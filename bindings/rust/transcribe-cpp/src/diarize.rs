//! [`DiarizeSession`] — the DIARIZE role (who spoke when), from [`Model::diarize_session`].
//! Threading, lifetime, and compute-lock rules are those of [`Session`](crate::Session).

use std::mem::ManuallyDrop;
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use transcribe_cpp_sys as sys;

use crate::cancel::{abort_trampoline, CancelToken};
use crate::error::{check, Error, Result};
use crate::family::{DiarizeExtension, DiarizeStreamExtension};
use crate::model::{Model, ModelInner};
use crate::result::{SpeakerSegment, Timings};
use crate::session::clamp_len;
use crate::streaming::StreamUpdate;
use crate::types::StreamState;

/// Static facts about a diarization model ([`Model::diarize_info`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DiarizeInfo {
    /// Input PCM rate (16000).
    pub sample_rate: i32,
    /// `speaker_id` is in `[1, max_speakers]`.
    pub max_speakers: i32,
}

/// Options for creating a diarization session.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DiarizeSessionOptions {
    /// CPU threads for ops that run on CPU; 0 = library default.
    pub n_threads: i32,
}

/// Per-run diarization parameters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DiarizeOptions {
    /// Optional family-specific extension (e.g. the Sortformer preset).
    pub family: Option<DiarizeExtension>,
}

/// Push-audio live diarization parameters ([`DiarizeSession::stream`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DiarizeStreamOptions {
    /// Optional family-specific extension (e.g. the Sortformer live preset).
    pub family: Option<DiarizeStreamExtension>,
}

/// A diarization session.
pub struct DiarizeSession {
    ptr: *mut sys::transcribe_diarize_session,
    // Keeps the native model alive and carries the per-model compute lock.
    model: Arc<ModelInner>,
    // Keeps the abort callback's userdata alive while installed.
    cancel: Option<Arc<AtomicBool>>,
}

impl std::fmt::Debug for DiarizeSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiarizeSession").finish_non_exhaustive()
    }
}

// SAFETY: as for `Session` — `&mut self` on the mutating calls keeps use to
// one thread at a time; deliberately NOT Sync.
unsafe impl Send for DiarizeSession {}

impl Drop for DiarizeSession {
    fn drop(&mut self) {
        unsafe { sys::transcribe_diarize_session_free(self.ptr) };
    }
}

impl DiarizeSession {
    pub(crate) fn new(model: &Model, options: &DiarizeSessionOptions) -> Result<DiarizeSession> {
        let mut params: sys::transcribe_diarize_session_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_diarize_session_params_init(&mut params) };
        params.n_threads = options.n_threads;

        let mut out: *mut sys::transcribe_diarize_session = std::ptr::null_mut();
        let status =
            unsafe { sys::transcribe_diarize_session_init(model.inner.ptr, &params, &mut out) };
        check(status, "diarize session init")?;
        debug_assert!(!out.is_null());

        Ok(DiarizeSession {
            ptr: out,
            model: Arc::clone(&model.inner),
            cancel: None,
        })
    }

    /// Install a [`CancelToken`] so an in-flight run can be aborted from
    /// another thread (the run then returns [`Error::Aborted`](crate::Error)).
    /// Replaces any previously installed token.
    pub fn set_cancel_token(&mut self, token: &CancelToken) {
        let flag = Arc::clone(&token.flag);
        let userdata = Arc::as_ptr(&flag) as *mut c_void;
        unsafe {
            sys::transcribe_diarize_set_abort_callback(self.ptr, Some(abort_trampoline), userdata)
        };
        self.cancel = Some(flag);
    }

    /// Remove any installed cancel token.
    pub fn clear_cancel_token(&mut self) {
        unsafe { sys::transcribe_diarize_set_abort_callback(self.ptr, None, std::ptr::null_mut()) };
        self.cancel = None;
    }

    /// Diarize one recording of 16 kHz mono float32 PCM. Returns the speaker
    /// segments, grouped by speaker and time-ordered within a speaker;
    /// segments of different speakers may overlap.
    pub fn run(&mut self, pcm: &[f32], options: &DiarizeOptions) -> Result<Vec<SpeakerSegment>> {
        let n = clamp_len(pcm.len())?;
        let family = options.family.as_ref().map(DiarizeExtension::materialize);
        let mut params: sys::transcribe_diarize_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_diarize_params_init(&mut params) };
        params.family = family.as_ref().map_or(std::ptr::null(), |f| f.ext_ptr());

        let status = self.model.with_compute(
            Some("a stream is active on this model; finish or drop it before diarize run()"),
            |_| unsafe { sys::transcribe_diarize_run(self.ptr, pcm.as_ptr(), n, &params) },
        )?;
        check(status, "diarize run")?;
        Ok(self.segments())
    }

    fn segments(&self) -> Vec<SpeakerSegment> {
        let count = unsafe { sys::transcribe_diarize_n_segments(self.ptr) };
        (0..count)
            .map(|i| {
                let mut raw: sys::transcribe_speaker_segment = unsafe { std::mem::zeroed() };
                unsafe { sys::transcribe_speaker_segment_init(&mut raw) };
                let _ = unsafe { sys::transcribe_diarize_get_segment(self.ptr, i, &mut raw) };
                SpeakerSegment::from_raw(&raw)
            })
            .collect()
    }

    /// Begin push-audio live diarization (models that accept a kind on
    /// [`ExtSlot::DiarizeStream`](crate::ExtSlot), e.g. Nemotron-3
    /// Diarization), returning a [`DiarizeStream`] that borrows this session.
    /// The stream holds the model's compute lease until it is finalized,
    /// reset or dropped; dropping it abandons the stream.
    pub fn stream(&mut self, options: &DiarizeStreamOptions) -> Result<DiarizeStream<'_>> {
        let family = options
            .family
            .as_ref()
            .map(DiarizeStreamExtension::materialize);
        let mut params: sys::transcribe_diarize_stream_params = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_diarize_stream_params_init(&mut params) };
        params.family = family.as_ref().map_or(std::ptr::null(), |f| f.ext_ptr());
        let ptr = self.ptr;
        self.model.with_compute(
            Some("a stream is already active on this model"),
            |lease| -> Result<()> {
                check(
                    unsafe { sys::transcribe_diarize_stream_begin(ptr, &params) },
                    "diarize stream begin",
                )?;
                *lease = true; // released at finalize/reset, or by drop
                Ok(())
            },
        )??;
        Ok(DiarizeStream {
            session: self,
            holds_lease: true,
        })
    }

    /// Begin a live stream that takes ownership of this session, for callers
    /// that keep it across calls (e.g. in a struct field). Same semantics as
    /// [`DiarizeSession::stream`]; get the session back with
    /// [`OwnedDiarizeStream::into_session`]. On error the session is returned
    /// unchanged alongside the error.
    pub fn into_stream(
        self,
        options: &DiarizeStreamOptions,
    ) -> std::result::Result<OwnedDiarizeStream, (Error, DiarizeSession)> {
        let session = NonNull::from(Box::leak(Box::new(self)));
        // SAFETY: the leaked session is only reachable through the stream
        // until `into_session` / `Drop` reclaims it, after the stream is gone.
        match unsafe { &mut *session.as_ptr() }.stream(options) {
            Ok(stream) => Ok(OwnedDiarizeStream {
                stream: ManuallyDrop::new(stream),
                session,
            }),
            // SAFETY: a failed `stream` call leaves no borrow behind.
            Err(err) => Err((err, *unsafe { Box::from_raw(session.as_ptr()) })),
        }
    }

    /// Model load time plus the last run's stage timings (`decode_ms` is 0).
    pub fn timings(&self) -> Timings {
        let mut raw: sys::transcribe_timings = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_timings_init(&mut raw) };
        let _ = unsafe { sys::transcribe_diarize_get_timings(self.ptr, &mut raw) };
        Timings::from_raw(&raw)
    }
}

/// An active live diarization stream borrowing its [`DiarizeSession`].
pub struct DiarizeStream<'a> {
    session: &'a mut DiarizeSession,
    // True while this stream holds the model's compute lease (see
    // `Stream` in session.rs).
    holds_lease: bool,
}

impl std::fmt::Debug for DiarizeStream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiarizeStream")
            .field("state", &self.state())
            .finish()
    }
}

impl Drop for DiarizeStream<'_> {
    fn drop(&mut self) {
        self.reset()
    }
}

impl DiarizeStream<'_> {
    /// Feed 16 kHz mono float32 PCM of any length; read the rows with
    /// [`segments`](Self::segments).
    pub fn feed(&mut self, pcm: &[f32]) -> Result<StreamUpdate> {
        let n = clamp_len(pcm.len())?;
        let mut update: sys::transcribe_stream_update = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_stream_update_init(&mut update) };
        let ptr = self.session.ptr;
        let held = self.holds_lease;
        let (status, ended) = self.session.model.with_compute(None, |lease| {
            let st =
                unsafe { sys::transcribe_diarize_stream_feed(ptr, pcm.as_ptr(), n, &mut update) };
            let ended = unsafe { sys::transcribe_diarize_stream_get_state(ptr) }
                == sys::transcribe_stream_state::TRANSCRIBE_STREAM_FAILED;
            if held && ended {
                *lease = false;
            }
            (st, ended)
        })?;
        if ended {
            self.holds_lease = false;
        }
        check(status, "diarize stream feed")?;
        Ok(StreamUpdate::from_raw(&update))
    }

    /// Flush the buffered tail; every row is final afterwards.
    pub fn finalize(&mut self) -> Result<StreamUpdate> {
        let mut update: sys::transcribe_stream_update = unsafe { std::mem::zeroed() };
        unsafe { sys::transcribe_stream_update_init(&mut update) };
        let ptr = self.session.ptr;
        let held = self.holds_lease;
        let status = self.session.model.with_compute(None, |lease| {
            let st = unsafe { sys::transcribe_diarize_stream_finalize(ptr, &mut update) };
            if held {
                *lease = false;
            }
            st
        })?;
        self.holds_lease = false;
        check(status, "diarize stream finalize")?;
        Ok(StreamUpdate::from_raw(&update))
    }

    /// Abandon the stream and clear its rows; releases the compute lease.
    pub fn reset(&mut self) {
        let ptr = self.session.ptr;
        let held = self.holds_lease;
        let _ = self.session.model.with_compute(None, |lease| {
            unsafe { sys::transcribe_diarize_stream_reset(ptr) };
            if held {
                *lease = false;
            }
        });
        self.holds_lease = false;
    }

    /// The speaker rows so far, grouped by speaker and time-ordered within a
    /// speaker (see [`SortformerLiveOptions`](crate::SortformerLiveOptions)
    /// for which rows are still open).
    pub fn segments(&self) -> Vec<SpeakerSegment> {
        self.session.segments()
    }

    /// The stream's lifecycle state.
    pub fn state(&self) -> StreamState {
        StreamState::from_raw(unsafe { sys::transcribe_diarize_stream_get_state(self.session.ptr) })
    }
}

/// A live diarization stream that owns its [`DiarizeSession`]. Created by
/// [`DiarizeSession::into_stream`]; otherwise identical to [`DiarizeStream`].
/// Dropping it abandons the stream and frees the session.
pub struct OwnedDiarizeStream {
    // Borrows `*session`, so it is always dropped first.
    stream: ManuallyDrop<DiarizeStream<'static>>,
    session: NonNull<DiarizeSession>,
}

// SAFETY: the stream is the only user of the heap session and both move
// together; `DiarizeStream` and `DiarizeSession` are each `Send`.
unsafe impl Send for OwnedDiarizeStream {}

impl std::fmt::Debug for OwnedDiarizeStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.stream.fmt(f)
    }
}

impl Drop for OwnedDiarizeStream {
    fn drop(&mut self) {
        // SAFETY: the stream is dropped before the session it borrows.
        unsafe {
            ManuallyDrop::drop(&mut self.stream);
            drop(Box::from_raw(self.session.as_ptr()));
        }
    }
}

impl OwnedDiarizeStream {
    /// See [`DiarizeStream::feed`].
    pub fn feed(&mut self, pcm: &[f32]) -> Result<StreamUpdate> {
        self.stream.feed(pcm)
    }

    /// See [`DiarizeStream::finalize`].
    pub fn finalize(&mut self) -> Result<StreamUpdate> {
        self.stream.finalize()
    }

    /// See [`DiarizeStream::reset`].
    pub fn reset(&mut self) {
        self.stream.reset()
    }

    /// See [`DiarizeStream::segments`].
    pub fn segments(&self) -> Vec<SpeakerSegment> {
        self.stream.segments()
    }

    /// See [`DiarizeStream::state`].
    pub fn state(&self) -> StreamState {
        self.stream.state()
    }

    /// Abandon the stream and return the session, idle and reusable.
    pub fn into_session(self) -> DiarizeSession {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: as in `Drop`, but the session is moved out instead of freed.
        unsafe {
            ManuallyDrop::drop(&mut this.stream);
            *Box::from_raw(this.session.as_ptr())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::sync::Mutex;

    #[test]
    fn run_is_busy_while_a_stream_holds_the_lease() {
        // Null native handles (both frees are NULL no-ops): the Busy check runs
        // under the lock before any native call.
        let mut session = DiarizeSession {
            ptr: std::ptr::null_mut(),
            model: Arc::new(ModelInner {
                ptr: std::ptr::null_mut(),
                compute_lock: Mutex::new(true),
            }),
            cancel: None,
        };
        let err = session
            .run(&[0.0; 160], &DiarizeOptions::default())
            .unwrap_err();
        let Error::Busy(msg) = err else {
            panic!("expected Busy, got {err:?}");
        };
        assert_eq!(
            msg,
            "a stream is active on this model; finish or drop it before diarize run()"
        );
    }
}
