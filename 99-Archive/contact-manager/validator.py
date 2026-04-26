import re


def validate_name(name):
    if not name.strip():
        print("이름을 입력해주세요.")
        return False
    return True


def validate_phone(phone):
    if not re.match(r'^010-\d{4}-\d{4}$', phone):
        print("전화번호 형식이 올바르지 않습니다. (예: 010-1234-5678)")
        return False
    return True


def validate_email(email):
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        print("이메일 형식이 올바르지 않습니다. (예: example@gmail.com)")
        return False
    return True
