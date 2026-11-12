# BCPR 소프트웨어 등록 자료 초안

## 프로그램 식별

- 프로그램명(국문): MOTIP 기반 BCPR 다중 객체 추적 소프트웨어
- 프로그램명(영문): MOTIP-BCPR: Observation-wise Attention Refinement for Multi-Object Tracking
- 프로그램 종류: 인공지능 영상 분석 / 다중 객체 추적 연구용 소프트웨어
- 기반 소프트웨어: MOTIP (Multiple Object Tracking as ID Prediction)

신청인, 개발자, 저작권자, 창작일 및 공표일은 실제 증빙과 일치하도록 신청자가
최종 기입해야 한다.

## 프로그램 개요

본 소프트웨어는 Transformer 기반 다중 객체 추적 모델 MOTIP의 ID association
과정에 BCPR observation-wise attention refinement를 추가한다. 현재 query와 causal
historical observations 및 competitor context의 관계를 이용해 관측별 attention bias를
생성하고, 동일한 입력과 정답을 공유하는 OFF/ON 두 경로의 ID loss로 학습한다.

## 주요 기능

1. DanceTrack 및 BFT 형식의 학습 데이터 로드
2. dataset-specific pretrained detector 로드
3. BCPR observation-wise bias 계산
4. causal/padding mask를 보존한 MOTIP cross-attention 주입
5. OFF/ON dual-path identity objective 기반 분산 학습
6. 체크포인트 저장·재개 및 α 조절 추론
7. TrackEval 기반 tracking 결과 생성 및 성능 평가
8. BCPR 구조·gradient·decoder integration 회귀 테스트

## BCPR 핵심 구현 파일

- `models/motip/bcpr.py`
- `models/motip/bcpr_base.py`
- `models/motip/id_decoder.py`
- `models/motip/trajectory_modeling.py`
- `models/motip/__init__.py`
- `train.py`
- `runtime_option.py`
- `configs/bcpr_*.yaml`
- `tests/test_bcpr_core.py`
- `tests/test_bcpr_decoder.py`

## 실행 방법

설치, 데이터 배치, 학습, 평가 및 체크포인트 사용법은 `docs/BCPR.md`에 기록되어
있다. 소스 패키지에는 개인정보, 데이터셋, pretrained weights, 실험 checkpoint 및
학습 output을 포함하지 않는다.

## 제3자 코드 및 라이선스

본 프로그램은 공식 MOTIP 저장소를 기반으로 하며, 원저작자 표시와 저장소의
`LICENSE`를 유지한다. BCPR 고유 변경 범위는 `NOTICE` 및 Git diff로 식별할 수 있다.
TrackEval과 Deformable DETR 등 상위 프로젝트가 사용하는 제3자 구성요소의 고지는
각 디렉터리와 원 프로젝트 문서를 따른다.
