from users.models import User

User.objects.create_superuser(
    username='admin_2',
    email='p_k7@mail.ru',
    password='YourSecurePassword123',
    first_name='Иван',
    second_name='Иванов',
    patronomyc='Иванович',
)
