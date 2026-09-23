# 정리 범위와 복원

기준 브랜치는 `codex/fix-f1tenth-rl-runtime`, 기준 커밋은 `abae63fd6c35800f80174f0e998b8960df76e7f2`입니다. 정리 브랜치 `codex/rl-driving-handoff`는 저장된 2156차원 SB3 PPO 모델로 실차를 구동하는 용도입니다.

원본 브랜치와 `main`의 파일을 삭제하거나 Git 기록을 다시 쓰지 않습니다. 제외한 자료는 기준 커밋에 남아 있으며, 활성 작업 공간에 대형 `archive/` 폴더를 추가하지 않습니다.

## 제외한 자료

| 경로 | 제외 근거 |
| --- | --- |
| 루트 `models/` | 과거 DQN `.pt` 체크포인트. PPO 주행 모델은 `src/f1tenth_rl/best_model.zip` |
| `src/mpc/`, `src/pure_pursuit/`, `src/stanley2_avoidance/` | 다른 제어 방식 |
| `src/launchers/` | 위 제어 방식용 시작 명령. PPO launch는 RL 패키지에 있음 |
| `src/waypoint_generator/` | 기본 지도 없는 PPO 입력에는 경로점 불필요 |
| `src/particle_filter/`, `range_libc/` | 기본 추론은 파티클 필터를 기다리지 않음. 차량 `/odom` 공급 코드는 유지 |
| `maps/`, `trash map/`, 루트 지도 파일 | 기본 모델은 지도·중심선 CSV 없이 동작하도록 설정됨 |
| `src/rplidar_ros`, `find_lidar_zero.py` | 현재 bringup은 Hokuyo `urg_node` 사용. 기존 방향 조사 스크립트는 RPLIDAR용 참고 코드 |
| `src/f1tenth_stack/` | package.xml 없는 별도 설정 복사본. 실제 패키지는 `src/f1tenth_system/f1tenth_stack/` |
| `drive_log/`, `pf_run/`, `rosbag2_*`, `driving_log.txt`, `frames.*` | 과거 실험·진단 결과 |
| `CMakeFiles/`, `*.whl`, `=1.22.0`, `=1.7.3`, 이름이 공백인 파일, `base_link:`, `laser:` | 빌드·설치 잔여물 및 잘못 생성된 파일 |
| 루트 `joy_teleop.yaml`, `Commands`, `current_env.txt`, 두 개의 `.backup`/`.save` 파일 | 설정 복사본, 옛 경로 중심 안내, 셸 환경 덤프, 임시 백업 |

371개 추적 항목을 제외했습니다. Git tree의 파일 크기 합은 원본 **386,131,725 bytes**, 제외 항목 **383,493,680 bytes**, 문서·launch 추가 전 유지 파일 **2,638,045 bytes**입니다. 디렉터리는 세지 않으며 삭제된 RPLidar gitlink의 크기는 0으로 계산했습니다.

이는 **작업 트리 파일 크기**입니다. 과거 Git 기록을 지우지 않았으므로 기존 `.git` 용량이 같은 비율로 줄어드는 것은 아닙니다. 후임에게 현 브랜치의 얕은 clone을 안내합니다.

## 옮긴 자료

| 기존 | 새 위치 |
| --- | --- |
| `current_pip_packages.txt` | `docs/environment/pip-packages-historical.txt` |
| `ros_packages.txt` | `docs/environment/ros-packages-historical.txt` |

## 폴더 정리에 수반되는 변경

- 모델 ZIP을 `f1tenth_rl`의 설치 데이터에 등록합니다. 원본 모델 파일 위치·내용은 유지합니다.
- `rl_drive_launch.py`는 설치된 모델로 `sb3_ppo` 추론을 실행합니다. 기존 범용 launch는 유지합니다.
- `f1tenth_stack/package.xml`에 이미 bringup에서 쓰는 `simple_teleop` 의존성을 명시합니다.
- RL 패키지에 launch용 의존성을 명시합니다.
- 생성되는 체크포인트·로그·설치 잔여물의 ignore 규칙을 추가합니다.

핵심 추론 노드, 모델 ZIP, `agent_params.yaml`, bringup, 센서·모터·조종기·mux 설정, 안전 노드 코드를 원본과 동일하게 보존합니다. 안전 노드의 현재 연결 상태는 [인수인계 안내](HANDOFF.md)를 참고합니다.

## 제외한 폴더가 필요할 때

현재 작업 변경을 커밋하거나 별도로 보관한 뒤 기준 커밋에서 특정 폴더만 복원할 수 있습니다. 다음은 MPC 관련 자료를 조사할 때의 예시입니다.

```bash
git fetch origin abae63fd6c35800f80174f0e998b8960df76e7f2
git restore --source=abae63fd6c35800f80174f0e998b8960df76e7f2 --worktree -- \
  src/mpc src/launchers src/waypoint_generator
```

다른 제어 방식을 실제 실행하려면 지도·위치 추정·Python 의존성과 실행 명령도 함께 검토합니다. 폴더 복원만으로 전체 실행 환경이 갖춰지는 것은 아닙니다.

## 검증 범위

아래 정적 검사를 통과했습니다. 차량에 연결하거나 모델을 실행하는 검사는 아닙니다.

- 핵심 모델·설정·실행 코드 9개 파일의 SHA-256이 기준 커밋과 동일
- 남은 Python 파일 79개의 Python 3.8 문법 파싱
- RL 노드의 패키지 내부 import 대상 존재 확인
- 모델 ZIP·신규 launch의 설치 데이터 등록과 주 실행 진입점 확인
- bringup이 참조하는 로컬 패키지 보존 및 외부 `joy`, `tf2_ros`, `urg_node` 의존성 선언 확인
- 추가·수정 파일의 공백 오류 검사
- 모델 ZIP의 JSON/텍스트 메타데이터 확인

ROS 전체 빌드·추론·실차 주행은 차량 환경에서 추가 확인해야 합니다. 확인되지 않은 성공 결과나 성능 수치는 기재하지 않습니다.
