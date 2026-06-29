import sys
import string, random


# Other Utils

def get_p_key(*lengths):
    """
    PK 생성 함수
    입력 : 각 구간의 문자열 길이 (*lengths)
    출력 : 랜덤 PK 문자열 (예: abc123-def456)
    """
    letters = string.ascii_lowercase + '0123456789'
    return '-'.join(''.join(random.choice(letters) for _ in range(l)) for l in lengths)
