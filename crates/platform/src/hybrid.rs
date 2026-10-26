//! [`HybridDecoder`]: a hardware decoder that turns into the software decoder for the same stream
//! when it fails mid-stream (decode error, session lost to a GPU change or sleep, an in-band
//! parameter-set change it was not set up for), without the caller noticing.
//!
//! The hybrid keeps the samples fed since the last point decoding can restart from (the run's
//! first sample, then every IDR / IRAP access unit). On a failure it builds the software decoder
//! ([`filmcraft_codecs::software_video_decoder`], skipping registered factories), replays those
//! samples through it and carries on, dropping pictures the hardware already returned and keeping
//! the ones it had decoded but not yet returned, so the output is the software decoder's own
//! sequence. That replay log is bounded; past the bound a failure is reported as an error once
//! and the instance continues in software from the next seek ([`VideoDecoder::reset`]).

use std::collections::{BTreeMap, BTreeSet};

use filmcraft_codecs::hw::StreamInfo;
use filmcraft_codecs::{CodecError, DecodedFrame, Result, VideoDecoder};
use filmcraft_isobmff::SampleEntry;

/// Replay log bounds: samples and bytes since the last restart point.
const LOG_MAX_SAMPLES: usize = 600;
const LOG_MAX_BYTES: usize = 256 << 20;
/// Presentation times remembered as returned (older ones are forgotten first).
const EMITTED_MAX: usize = 4096;
/// Pictures held while software replays a hardware run.  The replay log is bounded by bytes and
/// samples, but an invalid stream can still make one sample produce many decoded pictures.
const REPLAY_OUTPUT_MAX: usize = 4096;

/// Bounded restart input; submission alone never retires unreturned pictures.
#[derive(Default)]
struct ReplayLog {
    samples: Vec<(Vec<u8>, i64)>,
    bytes: usize,
    retire_before: usize,
}

impl ReplayLog {
    fn remember(&mut self, sample: &[u8], pts: i64, info: &StreamInfo, emitted: &BTreeSet<i64>) -> bool {
        // An individual sample cannot fit after any retirement; reject before walking hostile NAL lengths.
        if sample.len() > LOG_MAX_BYTES {
            return false;
        }
        if info.is_irap(sample) {
            // CRA leading pictures need the previous restart's references even after prior output.
            self.retire_before = if info.keeps_previous_restart(sample) {
                self.samples.iter().rposition(|(data, _)| info.is_irap(data)).unwrap_or(0)
            } else {
                self.samples.len()
            };
        }
        self.retire(emitted);
        let bytes = self.bytes.saturating_add(sample.len());
        if self.samples.len() >= LOG_MAX_SAMPLES || bytes > LOG_MAX_BYTES {
            return false;
        }
        self.samples.push((sample.to_vec(), pts));
        self.bytes = bytes;
        true
    }

    fn retire(&mut self, emitted: &BTreeSet<i64>) {
        let end = self.retire_before.min(self.samples.len());
        if end != 0 && self.samples[..end].iter().all(|(_, pts)| emitted.contains(pts)) {
            self.samples.drain(..end);
            self.bytes = self.samples.iter().map(|(data, _)| data.len()).sum();
            self.retire_before = 0;
        }
    }
}

/// A hardware decoder with a transparent software fallback.
pub struct HybridDecoder {
    hw: Option<Box<dyn VideoDecoder>>,
    sw: Option<Box<dyn VideoDecoder>>,
    entry: SampleEntry,
    info: StreamInfo,
    draft: bool,
    /// Restart input covering unreturned pictures; None disables replay until reset after a bound.
    log: Option<ReplayLog>,
    /// Pictures returned since the last reset (by pts).
    emitted: BTreeSet<i64>,
    /// Pictures the hardware had decoded but not returned when it failed.
    carry: BTreeMap<i64, DecodedFrame>,
}

impl HybridDecoder {
    /// Wrap `hw`, a decoder for `entry` (whose stream is described by `info`).
    pub fn new(hw: Box<dyn VideoDecoder>, entry: SampleEntry, info: impl Into<StreamInfo>) -> Self {
        let info = info.into();
        filmcraft_codecs::hw::note_hw_session();
        Self { hw: Some(hw), sw: None, entry, info, draft: false, log: Some(ReplayLog::default()), emitted: BTreeSet::new(), carry: BTreeMap::new() }
    }

    /// Whether the hardware decoder is still in use (false after a fallback).
    pub fn is_hardware(&self) -> bool {
        self.hw.is_some()
    }

    fn remember(&mut self, sample: &[u8], pts: i64) {
        let Some(log) = self.log.as_mut() else { return };
        if !log.remember(sample, pts, &self.info, &self.emitted) {
            self.log = None;
        }
    }

    fn note_emitted(&mut self, out: &[DecodedFrame]) {
        for f in out {
            self.emitted.insert(f.pts);
        }
        while self.emitted.len() > EMITTED_MAX {
            self.emitted.pop_first();
        }
        if let Some(log) = &mut self.log {
            log.retire(&self.emitted);
        }
    }

    /// Retain software replay output without letting a corrupt access unit turn the bounded
    /// sample replay into an unbounded frame queue.
    fn retain_replay(frames: &mut Vec<DecodedFrame>, mut out: Vec<DecodedFrame>) -> Result<()> {
        if frames.len().saturating_add(out.len()) > REPLAY_OUTPUT_MAX {
            return Err(CodecError::Decode("software fallback replay produced too many frames".into()));
        }
        frames.append(&mut out);
        Ok(())
    }

    /// In-band parameter sets (sequence header, key frame format) that differ from the sample
    /// entry's (the hardware session was set up from those).
    fn parameter_sets_changed(&self, sample: &[u8]) -> bool {
        self.info.parameters_changed(sample)
    }

    /// Software output after a fallback: pictures already returned are dropped, carried hardware
    /// pictures are merged in presentation order (a software picture replaces the carried one with
    /// the same pts).
    fn merge(&mut self, frames: Vec<DecodedFrame>) -> Vec<DecodedFrame> {
        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            if self.emitted.contains(&f.pts) {
                self.carry.remove(&f.pts);
                continue;
            }
            while let Some(e) = self.carry.first_entry() {
                if *e.key() >= f.pts {
                    break;
                }
                out.push(e.remove());
            }
            self.carry.remove(&f.pts);
            out.push(f);
        }
        self.note_emitted(&out);
        out
    }

    /// Salvage ready hardware pictures after a decode error, then replay in software.
    fn fall_back(&mut self, err: CodecError) -> Result<Vec<DecodedFrame>> {
        let Some(mut hw) = self.hw.take() else { return Err(err) };
        let drained = match hw.flush() {
            Ok(out) => out,
            Err(error) => {
                log::warn!("{} drain failed ({error}); replaying in software", hw.name());
                Vec::new()
            }
        };
        self.replay(hw, err, drained)
    }

    /// Replay buffered samples; retained hardware pictures cover damaged samples already decoded.
    fn replay(&mut self, hw: Box<dyn VideoDecoder>, err: CodecError, drained: Vec<DecodedFrame>) -> Result<Vec<DecodedFrame>> {
        log::warn!("{} failed ({err}); continuing with the software decoder", hw.name());
        filmcraft_codecs::hw::note_hw_fallback();
        for f in drained {
            if !self.emitted.contains(&f.pts) {
                self.carry.insert(f.pts, f);
            }
        }
        drop(hw);
        let mut sw = filmcraft_codecs::software_video_decoder(&self.entry).map_err(|e| CodecError::Decode(format!("{err}; no software decoder: {e}")))?;
        sw.set_draft(self.draft);
        self.sw = Some(sw);
        let Some(log) = self.log.take() else {
            // Too far from a restart point to replay: the next seek restarts in software.
            self.carry.clear();
            return Err(CodecError::Decode(format!("{err} (continuing in software after the next seek)")));
        };
        let mut frames = Vec::new();
        let mut rejected = None;
        let last = log.samples.len().saturating_sub(1);
        for (i, (s, p)) in log.samples.iter().enumerate() {
            let Some(sw) = self.sw.as_mut() else { break };
            match sw.decode(s, *p) {
                Ok(out) => Self::retain_replay(&mut frames, out)?,
                Err(e) if i == last => rejected = Some(e),
                Err(e) => log::warn!("software replay after the hardware decoder failed: {e}"),
            }
        }
        self.log = Some(log);
        match rejected {
            None => Ok(self.merge(frames)),
            Some(e) => {
                for f in frames {
                    if !self.emitted.contains(&f.pts) {
                        self.carry.insert(f.pts, f);
                    }
                }
                Err(e)
            }
        }
    }
}

impl VideoDecoder for HybridDecoder {
    fn decode(&mut self, sample: &[u8], pts: i64) -> Result<Vec<DecodedFrame>> {
        if let Some(sw) = self.sw.as_mut() {
            let frames = sw.decode(sample, pts)?;
            return Ok(self.merge(frames));
        }
        self.remember(sample, pts);
        if self.parameter_sets_changed(sample) {
            return self.fall_back(CodecError::Decode("in-band parameter sets differ from the sample entry".into()));
        }
        let Some(hw) = self.hw.as_mut() else {
            return Err(CodecError::Decode("no decoder".into()));
        };
        match hw.decode(sample, pts) {
            Ok(out) => {
                filmcraft_codecs::hw::note_hw_frames(out.len());
                self.note_emitted(&out);
                Ok(out)
            }
            Err(e) => self.fall_back(e),
        }
    }

    fn flush(&mut self) -> filmcraft_codecs::Result<Vec<DecodedFrame>> {
        let mut out = Vec::new();
        if let Some(hw) = self.hw.as_mut() {
            match hw.flush() {
                Ok(frames) => {
                    filmcraft_codecs::hw::note_hw_frames(frames.len());
                    self.note_emitted(&frames);
                    return Ok(frames);
                }
                Err(error) => {
                    let Some(hw) = self.hw.take() else { return Err(error) };
                    // The failed drain published nothing; replay without draining hardware twice.
                    out = self.replay(hw, error, Vec::new())?;
                }
            }
        }
        let frames = self.sw.as_mut().ok_or_else(|| CodecError::Decode("no decoder".into()))?.flush()?;
        out.extend(self.merge(frames));
        out.extend(std::mem::take(&mut self.carry).into_values());
        Ok(out)
    }

    fn reset(&mut self) {
        if let Some(hw) = self.hw.as_mut() {
            hw.reset();
        }
        if let Some(sw) = self.sw.as_mut() {
            sw.reset();
        }
        self.log = Some(ReplayLog::default());
        self.emitted.clear();
        self.carry.clear();
    }

    fn name(&self) -> &str {
        match (&self.hw, &self.sw) {
            (Some(hw), _) => hw.name(),
            (None, Some(sw)) => sw.name(),
            (None, None) => "hardware decoder",
        }
    }

    fn is_random_access(&self, sample: &[u8]) -> Option<bool> {
        self.info.is_random_access(sample)
    }

    fn is_disposable(&self, sample: &[u8]) -> bool {
        self.info.is_disposable(sample)
    }

    fn set_draft(&mut self, on: bool) {
        self.draft = on;
        if let Some(sw) = self.sw.as_mut() {
            sw.set_draft(on);
        }
    }
}

#[cfg(test)]
mod tests {
    use filmcraft_frame::VideoFrame;

    use super::*;

    fn frame(pts: i64) -> DecodedFrame {
        DecodedFrame { pts, frame: VideoFrame::rgba8(1, 1, vec![0, 0, 0, 255]), draft: false }
    }

    fn nal_info(codec: filmcraft_codecs::hw::NalCodec) -> StreamInfo {
        // Header-only tests: one macroblock, 8-bit 4:2:0, square pixels; no decoder reads these samples.
        filmcraft_codecs::hw::NalStreamInfo {
            codec,
            length_size: 1,
            highest_tid: None,
            parameter_sets: Vec::new(),
            coded: (16, 16),
            crop: (0, 0, 16, 16),
            chroma_format_idc: 1,
            bit_depth_luma: 8,
            bit_depth_chroma: 8,
            interlaced: false,
            profile_idc: 1,
            color: filmcraft_color::ColorInfo::REC709,
            par: (1, 1),
            reorder: 0,
        }
        .into()
    }

    #[test]
    fn replay_prefix_waits_for_every_output_and_releases_bytes() {
        let payloads = [b"a".as_slice(), b"bb".as_slice(), b"ccc".as_slice()];
        let mut log = ReplayLog {
            samples: payloads.iter().enumerate().map(|(pts, data)| (data.to_vec(), pts as i64)).collect(),
            bytes: payloads.iter().map(|data| data.len()).sum(),
            retire_before: payloads.len() - 1,
        };
        let mut emitted = BTreeSet::from([0]);
        log.retire(&emitted);
        assert_eq!(log.samples.len(), payloads.len(), "one unreturned prefix picture prevents retirement");
        emitted.insert(1);
        log.retire(&emitted);
        assert_eq!(log.samples, vec![(payloads[2].to_vec(), 2)]);
        assert_eq!(log.bytes, payloads[2].len());
        assert_eq!(log.retire_before, 0, "restart index rebased with the retained sample");
    }

    #[test]
    fn restart_input_keeps_its_sample_and_byte_bounds() {
        let info = nal_info(filmcraft_codecs::hw::NalCodec::H264);
        // One-byte length prefix + H.264 IDR header (NAL type 5).
        let idr = [1, 0x65];
        let emitted = BTreeSet::new();
        let mut log = ReplayLog::default();
        for pts in 0..LOG_MAX_SAMPLES {
            assert!(log.remember(&idr, pts as i64, &info, &emitted));
        }
        assert!(!log.remember(&idr, LOG_MAX_SAMPLES as i64, &info, &emitted));
        assert_eq!(log.samples.len(), LOG_MAX_SAMPLES, "restart submission cannot discard unreturned input");
        let oversized = vec![0; LOG_MAX_BYTES + 1];
        let mut log = ReplayLog::default();
        assert!(!log.remember(&oversized, 0, &info, &emitted));
        assert!(log.samples.is_empty(), "oversized input rejected before copying");
        assert_eq!(log.bytes, 0);
    }

    #[test]
    fn cra_keeps_previous_restart_references_after_output() {
        let info = nal_info(filmcraft_codecs::hw::NalCodec::Hevc);
        // One-byte length + HEVC IDR_W_RADL (19) / CRA (21), temporal_id_plus1=1.
        let (idr, cra) = ([2, 19 << 1, 1], [2, 21 << 1, 1]);
        let mut log = ReplayLog::default();
        assert!(log.remember(&idr, 0, &info, &BTreeSet::new()));
        assert!(log.remember(&cra, 1, &info, &BTreeSet::from([0])));
        assert_eq!(log.samples, vec![(idr.to_vec(), 0), (cra.to_vec(), 1)]);
    }

    #[test]
    fn replay_output_has_a_hard_limit() {
        let mut retained = Vec::new();
        HybridDecoder::retain_replay(&mut retained, (0..REPLAY_OUTPUT_MAX as i64).map(frame).collect()).unwrap();

        let err = HybridDecoder::retain_replay(&mut retained, vec![frame(REPLAY_OUTPUT_MAX as i64)]).unwrap_err();
        assert!(err.to_string().contains("too many frames"));
        assert_eq!(retained.len(), REPLAY_OUTPUT_MAX);
    }
}
