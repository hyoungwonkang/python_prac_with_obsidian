class Contact:
    def __init__(self, name, phone, email, memo=""):
        self.name = name
        self.phone = phone
        self.email = email
        self.memo = memo

    def to_dict(self):
        return {
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "memo": self.memo,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            phone=data["phone"],
            email=data["email"],
            memo=data.get("memo", ""),
        )

    def __str__(self):
        info = f"이름: {self.name} | 전화: {self.phone} | 이메일: {self.email}"
        if self.memo:
            info += f" | 메모: {self.memo}"
        return info
