from __future__ import annotations
import numpy as np

from spikeinterface.core.core_tools import define_function_from_class

from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from spikeinterface.core.baserecording import BaseRecording

from spikeinterface.preprocessing.filter import fix_dtype


class PhaseSampling(BasePreprocessor):

    def __init__(
        self,
        recording: BaseRecording,
        dtype: str | np.dtype | None = None,
    ):
        dtype_ = fix_dtype(recording, dtype)
        BasePreprocessor.__init__(self, recording, dtype=dtype_)

        for parent_segment in recording._recording_segments:
            rec_segment = RemoveMechanicalNoiseRecordingSegment(parent_segment, dtype_)
            self.add_recording_segment(rec_segment)

        self._kwargs = dict(
            recording=recording,
            dtype=dtype_.str,
        )


class PhaseSamplingSegment(BasePreprocessorSegment):
    def __init__(
        self,
        parent_recording_segment,
        dtype,
    ):
        BasePreprocessorSegment.__init__(self, parent_recording_segment)

        self.dtype = dtype

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        ref_data = np.mean(traces, axis=1)
        mov_ccnoise = get_mech_noise(ref_data, self.mov_window, self.interval)
        idx_clean = np.where(mov_ccnoise < np.median(mov_ccnoise) * self.median_cap)[0]
        data_all_cleaned = np.zeros_like(traces)
        data_all_cleaned[idx_clean] = traces[idx_clean]
        return data_all_cleaned.astype(self.dtype, copy=False)


rm_mechanical_noise = define_function_from_class(source_class=PhaseSampling, name="rm_mechanical_noise")
