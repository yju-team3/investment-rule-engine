# investment-rule-engine

## 1. 전체 아키텍처 설계

- **decision_engine** 패키지는 입력 스냅샷(시장/종목/포트폴리오)을 받아 6단계 규칙 기반으로 결과를 생성한다.
- **Regime → Gate → Classification → Entry → Position → Final Output** 순서로 처리한다.
- 모든 규칙은 독립 클래스이며, 결과는 로그로 누적된다.

구성 요소:

- `decision_engine.models`: 입력/출력 데이터 구조와 열거형 정의
- `decision_engine.rules`: 레짐, 게이트, 분류, 진입, 비중 룰 정의
- `decision_engine.engine`: 규칙 실행 파이프라인
- `decision_engine.demo`: 샘플 종목 실행

## 2. 핵심 룰 엔진 구조

- **RegimeRule**: 단순 지표(지수 vs MA, VIX)로 레짐 판정.
- **GateRule**: 유동성/변동성/레짐 불일치/이벤트/비즈니스 명확성 등 즉시 탈락 조건 처리.
- **Classifier**: 후보 유형을 단 하나로 확정, 복수 충돌 시 보류.
- **EntryEvaluator**: 후보 유형별 진입 신호를 분리 처리.
- **PositionSizer**: 변동성 기반으로 최대 비중을 제한하고 분할 트랜치 제시.

## 3. 최소 실행 가능한 MVP

```bash
python -m decision_engine.demo
```

샘플 종목 2개에 대해 다음 출력 형식을 콘솔에 표시한다.

```
(1) Decision
APPROVE / WAIT / REJECT
(2) Reason Log
- 규칙 통과/보류/거절 로그
(3) Action Plan
- 1차 진입 조건
- 추가 진입 조건
- 비중 상한
- 무효화 조건
- 금지 사항
```

## 테스트

```bash
python -m unittest
```
