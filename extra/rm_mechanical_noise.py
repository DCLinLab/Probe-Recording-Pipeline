from __future__ import annotations
import numpy as np
from numba import njit, prange

from spikeinterface.core.core_tools import define_function_from_class

from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.baserecording import BaseRecording

from spikeinterface.preprocessing.filter import fix_dtype


class RemoveMechanicalNoiseRecording(BasePreprocessor):

    def __init__(
        self,
        recording: BaseRecording,
        mov_window=2000,
        interval=100,
        median_cap=2.5,
        dtype: str | np.dtype | None = None,
    ):
        dtype_ = fix_dtype(recording, dtype)
        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        for parent_segment in recording._recording_segments:
            rec_segment = RemoveMechanicalNoiseRecordingSegment(
                parent_segment, mov_window, interval, median_cap, dtype_
            )
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            mov_window=mov_window,
            interval=interval,
            median_cap=median_cap,
            dtype=dtype_.str,
        )


@njit(parallel=False, fastmath=True)
def get_mech_noise(data, mov_window, interval):
    mov_ccnoise = np.zeros_like(data)
    for j in prange(data.shape[0] // interval):
        i = j * interval
        idx0 = max(0, i + (interval - mov_window) // 2)
        idx1 = min(data.shape[0], i + (interval + mov_window) // 2)
        v = data[idx0: idx1]
        mov_ccnoise[i: i + interval] = np.correlate(v, v).max()
    mov_ccnoise = (mov_ccnoise - mov_ccnoise.min()) / (mov_ccnoise.max() - mov_ccnoise.min())
    return mov_ccnoise


class RemoveMechanicalNoiseRecordingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        mov_window,
        interval,
        median_cap,
        dtype,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.mov_window = mov_window
        self.interval = interval
        self.median_cap = median_cap
        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        ref_data = np.mean(traces, axis=1)
        mov_ccnoise = get_mech_noise(ref_data, self.mov_window, self.interval)
        idx_clean = np.where(mov_ccnoise < np.median(mov_ccnoise) * self.median_cap)[0]
        data_all_cleaned = np.zeros_like(traces)
        data_all_cleaned[idx_clean] = traces[idx_clean]
        return data_all_cleaned.astype(self.dtype, copy=False)


rm_mechanical_noise = define_function_from_class(source_class=RemoveMechanicalNoiseRecording, name="rm_mechanical_noise")
