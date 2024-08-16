# 실시간 한글 수화 번역기(Real-time-Korean-Sign-Language-Translator)

실시간으로 한글(자음,모음) 수화를 번역하여 청각장애인과 비장애인의 의사소통을 원활할 수 있도록 도와주는 실시간 수화 번역기

![image](https://github.com/user-attachments/assets/4eb95147-2cc6-4203-810c-2431e53dc24e)

- 기간: 2022.11 ~ 2022.12
- 역할: 팀장
- 기여도: 60%
- 사용한 기술 및 도구:
    - 언어: Python
    - IDE: PyCharm
    - 도구: Mediapipe
- 업무:
    - 한글 음절 조합 기능 개발
        - 유니코드 조합 공식인 유니코드 = 0xAC00 + (초성 인덱스 * 21 * 28) + (중성 인덱스 * 28) + 종성 인덱스 활용
    - 조합된 글자 혹은 문장 출력
