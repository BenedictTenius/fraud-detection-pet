import os
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils
from catboost import CatBoostClassifier


class TritonPythonModel:
    def initialize(self, args: dict[str, str]) -> None:
        model_path = Path(os.getenv("CATBOOST_MODEL_PATH", "/artifacts/catboost.cbm"))
        if not model_path.is_file():
            raise FileNotFoundError(f"CatBoost model is missing: {model_path}")

        instance_kind = args["model_instance_kind"]
        self._task_type = "GPU" if instance_kind == "GPU" else "CPU"
        if self._task_type == "GPU":
            os.environ["CUDA_VISIBLE_DEVICES"] = args["model_instance_device_id"]

        self._model = CatBoostClassifier().load_model(str(model_path))

    def execute(self, requests: list) -> list:
        try:
            batches = [self._input(request) for request in requests]
            sizes = [len(batch) for batch in batches]
            features = np.ascontiguousarray(np.concatenate(batches), dtype=np.float32)
            scores = self._model.predict(
                features,
                prediction_type="Probability",
                task_type=self._task_type,
            )[:, 1].astype(np.float32, copy=False)

            responses = []
            offset = 0
            for size in sizes:
                output = scores[offset : offset + size].reshape(-1, 1)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("output__0", output)]
                    )
                )
                offset += size
            return responses
        except Exception as error:
            message = pb_utils.TritonError(str(error))
            return [pb_utils.InferenceResponse(error=message) for _ in requests]

    @staticmethod
    def _input(request) -> np.ndarray:
        tensor = pb_utils.get_input_tensor_by_name(request, "input__0")
        if tensor is None:
            raise ValueError("input__0 is required")
        features = tensor.as_numpy()
        if features.ndim != 2 or features.shape[1] != 8:
            raise ValueError("input__0 must have shape [batch, 8]")
        return features
