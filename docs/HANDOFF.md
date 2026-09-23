# 후임 인수인계: PPO 실차 주행

## 실행 구조

저장된 PPO 정책이 라이다와 차량 속도를 받아 조향각·속도를 계산합니다. 기본 모델은 지도 없는 **2,156차원 입력**을 사용하므로 지도, 경로점 CSV, 파티클 필터가 필요하지 않습니다.

노드에는 `/pf/pose/odom` 구독과 학습 코드가 남아 있지만 `training_mode=false`의 SB3 추론은 그 메시지를 기다리지 않습니다. 실제 속도 입력은 VESC가 공급하는 `/odom`입니다. `/odom`까지 불필요하다고 판단해 제거하면 안 됩니다.

| 토픽 | 공급자 또는 소비자 |
| --- | --- |
| `/scan` | `urg_node`가 Hokuyo 라이다에서 발행 |
| `/odom` | `vesc_to_odom_node`가 차량 속도 등을 발행 |
| `/drive` | PPO 노드가 조향·속도를 발행 |
| `/teleop` | 조종기의 수동 주행 명령 |
| `/ackermann_drive` | `ackermann_mux`가 선택한 명령을 VESC 변환 노드에 전달 |

## 모델과 설정 위치

| 파일 | 용도 |
| --- | --- |
| `src/f1tenth_rl/best_model.zip` | 학습 완료 모델 |
| `src/f1tenth_rl/config/agent_params.yaml` | 입력 전처리, 조향·속도 보정 |
| `src/f1tenth_rl/f1tenth_rl/rl_agent_node.py` | 추론과 `/drive` 발행 |
| `src/f1tenth_system/f1tenth_stack/config/sensors.yaml` | Hokuyo IP·포트·스캔 범위 |
| `src/f1tenth_system/f1tenth_stack/config/vesc.yaml` | 모터 포트, 속도·조향 변환, `/odom` |
| `src/f1tenth_system/f1tenth_stack/config/joy_teleop.yaml` | 조종기 버튼과 축 |
| `src/f1tenth_system/f1tenth_stack/config/mux.yaml` | 수동·자율 명령 우선순위 |

별도 복사본 `src/f1tenth_stack/config/vesc.yaml`은 정리했습니다. 실제 bringup은 **`src/f1tenth_system/f1tenth_stack/` 아래 설정**을 읽습니다.

ZIP 내부 JSON/텍스트로 확인한 모델 정보입니다. 모델을 역직렬화하거나 실행해 확인한 결과는 아닙니다.

| 항목 | 값 |
| --- | --- |
| 입력 / 출력 크기 | 2156 / 2 |
| 저장된 학습 스텝 | 200000 |
| 모델 저장 당시 SB3 | 1.8.0 |
| 기록된 학습 환경 | Python 3.8.10, PyTorch 2.4.1+cu121, NumPy 1.24.4, Gym 0.19.0 |
| SHA-256 | `6888653d5fbc085348c808b3d421c2a80f4f10bbffc340a1ae351a09591fa48f` |

현재 설정의 입력은 `[라이다 2155개, 속도 / 3.2]`입니다. 라이다는 10 m까지 잘라 정규화하고 필요하면 2155개로 보간하며 `sb3_reverse_scan=true`를 적용합니다. 순서·스케일·좌우 반전은 학습 입력과 일치해야 합니다.

2160차원 모델이나 경로 기반 제어로 바꿀 때는 중심선·위치 추정 요구를 다시 검토합니다. 이 정리본의 지도 불필요 결론을 다른 모델에 그대로 적용하지 않습니다.

## 환경과 빌드

기존 차량의 작동 환경을 유지한 상태에서 별도 폴더로 받습니다.

```bash
git clone --depth 1 --single-branch --branch codex/rl-driving-handoff \
  https://github.com/jeongdo-choi/f1tenth_ws_rl.git f1tenth_drive_ws
cd f1tenth_drive_ws
source /opt/ros/foxy/setup.bash
rosdep check --from-paths src --ignore-src
```

`rosdep`이 초기화되어 있고 해당 Foxy 패키지 저장소를 사용할 수 있어야 합니다. 빠진 ROS 의존성은 기존 장비 환경에 맞춰 설치합니다.

```bash
rosdep install --from-paths src --ignore-src -r -y
```

`urg_node`, `joy`, ROS 메시지·TF·launch 패키지 등은 외부 의존성입니다. `rosdep`만으로 Python 모델 라이브러리가 전부 준비되지는 않습니다. 노드가 사용하는 Python에서 `numpy`, `torch`, `stable_baselines3`와 해당 버전의 Gym/Gymnasium 요구를 확인합니다. 학습 스크립트를 유지하므로 `setup.py`에는 학습용 의존성도 남아 있습니다.

**학습 환경과 차량의 추론 환경을 같다고 단정하지 않습니다.** `docs/environment/`의 과거 패키지 목록은 최신 의존성 잠금 파일이 아닙니다. 과거 차량 목록의 PyTorch 버전도 모델에 기록된 학습 버전과 다릅니다. 새 장비에 최신 라이브러리를 일괄 설치하기보다 기존 차량에서 실행되던 Python 환경과 SB3 모델 로딩 호환성을 먼저 확인합니다.

```bash
colcon build --symlink-install --packages-up-to f1tenth_stack f1tenth_rl safety_node
source install/local_setup.bash
ros2 pkg prefix f1tenth_rl
ros2 pkg prefix f1tenth_stack
```

출력이 새 작업 공간을 가리키는지 확인합니다. 예전 작업 공간을 자동 source하는 셸 설정도 확인합니다. 삭제한 옛 패키지가 `install/`에 남아 우연히 실행되는 일을 막기 위해 새 폴더에서 빌드합니다. 기존 차량 폴더를 자동 삭제하는 작업은 포함하지 않습니다.

## 장비 확인과 실행

보존된 설정은 Hokuyo `192.168.0.10:10940`, 모터 `/dev/ttyACM0`입니다. 장비 구성이 다르면 센서·모터 설정부터 확인합니다. 라이다를 중복 실행하지 않습니다.

터미널 1:

```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

모델을 실행하기 전에 `/scan`과 `/odom` 수신을 확인합니다.

```bash
ros2 topic hz /scan
# 확인 후 Ctrl+C
ros2 topic hz /odom
# 확인 후 Ctrl+C
```

조향 방향, 조종기 수동 전환과 정지 동작을 먼저 확인합니다. 현재 `joy_teleop.yaml`의 `autonomous_control`은 `/dev/null`로 메시지를 보내며 PPO 실행을 허가·차단하는 스위치로 연결되어 있지 않습니다. `safety_node`도 기본 bringup에서 실행하지 않고 mux의 전용 비상제동 항목은 주석 처리되어 있습니다. **안전 노드 폴더가 있다는 이유로 자동 비상제동이 켜져 있다고 판단하지 않습니다.** 이번 폴더 정리는 그 동작을 새로 구현하거나 바꾸지 않습니다.

터미널 2에서 다음 명령을 실행하면 주행 명령이 나갈 수 있습니다.

```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_drive_launch.py
```

새 launch는 모델을 패키지 설치 위치에서 찾아 `/home/meric/...`에 의존하지 않습니다. 로그의 입력 크기·전처리 설정도 확인합니다. 별도 모델·설정을 사용하려면 다음 인수를 지정합니다.

```bash
ros2 launch f1tenth_rl rl_drive_launch.py \
  model_path:=/absolute/path/to/best_model.zip \
  params_file:=/absolute/path/to/agent_params.yaml
```

기존 `rl_agent_launch.py`는 기본 인수가 YAML의 배포 설정을 덮어씁니다. 그 명령을 사용할 때는 `training_mode:=false model_type:=sb3_ppo model_path:=...`를 모두 명시합니다.

보존된 `drive_max_speed`는 2.50 m/s입니다. 첫 지상 시험 속도로 적절한지 현장에서 판단하고 필요하면 별도 설정 파일에서 낮춥니다. 기존 주행 코드·튜닝 보존이 새 정리본의 실차 검증을 대신하지 않습니다.

## 남겨 둔 학습 코드와 과거 기록

`f1tenth_rl/models/`, `environment.py`, `utils/rewards.py`는 노드가 직접 import하거나 생성합니다. 폴더만 삭제하면 시작 단계부터 실패할 수 있어 보존했습니다. `scripts/` 역시 등록된 학습·평가 실행 진입점이라 유지합니다. 이 코드가 저장된 SB3 모델의 원래 학습 파이프라인 전체를 재현한다는 뜻은 아닙니다.

- `docs/environment/pip-packages-historical.txt`, `ros-packages-historical.txt`: 과거 환경 조사 기록. 설치 스크립트가 아닙니다.
