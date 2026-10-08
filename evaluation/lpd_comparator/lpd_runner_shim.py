#!/usr/bin/env python3
"""AMENDMENT 4 — LPD 실행 환경 셔림 (동결 러너 바이트 무변경).

이 호스트의 ODL 0.8.2 는 `odl/contrib/torch/operator.py:243` 에서 정의된 적 없는
모듈 상수 `AVOID_UNNECESSARY_COPY` 를 참조한다(NameError). 원저자 본인의 레거시
스크립트 3곳(`examples/train_unet_fanbeam_shepp_50to1000.py` 등)이 동일한 런타임
패치를 이미 사용한다:

    if not hasattr(op, "AVOID_UNNECESSARY_COPY"):
        op.AVOID_UNNECESSARY_COPY = None if numpy_major >= 2 else True

이 값은 `np.stack(results).astype(dtype, copy=X)` 의 copy 플래그로만 쓰이므로
**수치 결과에 영향이 없다**(복사 여부만 결정). numpy 1.26.4 이므로 True 를 쓴다 —
원저자 패치와 동일한 값이다.

동결 LPD 러너(`lpd_sparseview_runner.py`, sha256 9392422…)는 한 바이트도 수정하지
않으며, 이 셔림이 속성을 주입한 뒤 러너를 __main__ 으로 그대로 실행한다.

사용:  python lpd_runner_shim.py <runner_path> <runner args...>
"""
import runpy
import sys

import numpy as np
import odl.contrib.torch.operator as _odl_torch_op

if not hasattr(_odl_torch_op, "AVOID_UNNECESSARY_COPY"):
    _odl_torch_op.AVOID_UNNECESSARY_COPY = (
        None if int(np.__version__.split(".")[0]) >= 2 else True
    )
    print(
        f"[shim] odl.contrib.torch.operator.AVOID_UNNECESSARY_COPY = "
        f"{_odl_torch_op.AVOID_UNNECESSARY_COPY} (numpy {np.__version__})",
        flush=True,
    )

runner = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(runner, run_name="__main__")
