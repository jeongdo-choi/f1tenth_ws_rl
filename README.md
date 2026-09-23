# Echo F1TENTH — 학습된 PPO 모델로 실차 주행

2025-Fall ~ 2026-Spring, Chonnam National University Mechanical Engineering / 전남대 기계공학부

Team **Echo** · 소상록(TL), 최정도, 강현우 · Adviser: Prof. SKo

이 브랜치는 **학습된 `best_model.zip`을 실제 차량에서 실행하는 후임 인수인계용 작업 공간**입니다. 기준은 `codex/fix-f1tenth-rl-runtime`의 `abae63fd6c35800f80174f0e998b8960df76e7f2`입니다. 기존 센서 설정, 모델, 추론 코드, 조향·속도 보정값은 유지합니다.

## 처음 볼 때

1. [인수인계 안내](docs/HANDOFF.md): 환경, 모델, 빌드·주행 명령, 설정 위치
2. [정리 내역과 복원 방법](docs/CLEANUP.md): 어떤 폴더를 왜 제외했는지

## 남긴 폴더

| 위치 | 역할 |
| --- | --- |
| `src/f1tenth_rl/` | PPO 모델, 추론 노드, 전처리·조향·감속 설정 |
| `src/f1tenth_system/` | Hokuyo 라이다, VESC 모터·조향·속도계, 조종기, 명령 중계 |
| `src/simple_teleop/` | 현재 차량 시작 명령에서 실행하는 조종기 노드 |
| `src/safety_node/` | 기존 TTC 비상제동 코드 보존. 기본 시작 명령에서 자동 실행되지는 않음 |
| `docs/` | 인수인계 문서와 과거 환경 기록 |

주행용 학습 결과는 **`src/f1tenth_rl/best_model.zip`**입니다. 이름이 비슷한 `src/f1tenth_rl/f1tenth_rl/models/`는 Python 코드이므로 지우면 안 됩니다.

## 실행 요약

기존 차량의 ROS 2 Foxy 및 Python 실행 환경이 준비되어 있다는 전제입니다. 새 작업 폴더에서 빌드하며, 예전 작업 공간의 `install/`을 함께 source하지 않습니다. 자세한 조건은 [인수인계 안내](docs/HANDOFF.md)를 따릅니다.

```bash
source /opt/ros/foxy/setup.bash
colcon build --symlink-install --packages-up-to f1tenth_stack f1tenth_rl safety_node
source install/local_setup.bash
```

터미널 1 — 센서·모터·조종기:

```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_stack bringup_launch.py
```

터미널 2 — 학습된 PPO 모델로 주행:

```bash
source /opt/ros/foxy/setup.bash
source install/local_setup.bash
ros2 launch f1tenth_rl rl_drive_launch.py
```

새 `rl_drive_launch.py`는 설치된 모델을 찾아 `training_mode=false`, `model_type=sb3_ppo`로 실행합니다. 기존 `rl_agent_launch.py`는 기본값이 학습 모드이므로 후임용 주행 명령으로 사용하지 않습니다.

**주행 명령을 실행하면 `/drive`가 발행될 수 있습니다.** 보존된 최대속도 설정은 2.50 m/s입니다. 자동 비상제동과 자율주행 허가 버튼이 기본 구성에 연결되어 있다고 가정하지 말고, 실제 차량에서 정지·수동 전환과 센서 방향을 확인한 뒤 지상 주행을 시작합니다.

## 검증 범위

파일 참조·패키징·Python 문법·모델 메타데이터와 보존 파일의 체크섬을 점검합니다. ROS 전체 빌드, 모델 추론 실행, 실제 차량 주행 성공은 별도 확인이 필요합니다. 기존 튜닝·모델 보존이 새 정리본의 실차 검증을 대신하지는 않습니다.
