from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
import re
from django.utils.translation import gettext_lazy as _
from django.db import models
from django.contrib.auth import get_user_model

class Role(models.Model):
    name = models.CharField(max_length=50, unique=True)
    
    def __str__(self):
        return self.name

class User(AbstractUser):
    role = models.ForeignKey(Role, on_delete=models.CASCADE, null=True, blank=True)
    first_name = models.CharField(max_length=80,default="")
    second_name = models.CharField(max_length=80,default="")
    patronomyc = models.CharField(max_length=80,default="")
    #birth_date = models.DateField(auto_now_add=True)
    phone = models.CharField(max_length=60,default="")
    email = models.EmailField()
    date_of_birth = models.DateField(null=True, blank=True)

    def clean(self):
        super().clean()
        self.validate_names()
        self.validate_phone()

    def validate_names(self):
        for name in [self.first_name, self.second_name, self.patronomyc]:
            if not name.istitle():
                raise ValidationError(_('Имя, фамилия и отчество должны начинаться с заглавной буквы.'))

    def validate_phone(self):
        phone_regex = r'^8-\d{3}-\d{3}-\d{2}-\d{2}$'
        if not re.match(phone_regex, self.phone):
            raise ValidationError(_('Телефон должен быть в формате 8-XXX-XXX-XX-XX.'))

    def save(self, *args, **kwargs):
        self.full_clean()  # Вызываем валидацию перед сохранением
        super().save(*args, **kwargs)

class Transactions(models.Model): 
    bin = models.CharField(max_length=100, verbose_name='bin')
    amount = models.CharField(max_length=100, verbose_name='amount')
    shoppercountrycode = models.CharField(max_length=100, verbose_name='shoppercountrycode')
    cardverificationcodesupplied = models.CharField(max_length=100, verbose_name='cardverificationcodesupplied')
    cvcresponsecode = models.CharField(max_length=100, verbose_name='cvcresponsecode')
    txvariantcode = models.CharField(max_length=100, verbose_name='txvariantcode')
    Day = models.CharField(max_length=100, verbose_name='Day')
    Month = models.CharField(max_length=100, verbose_name='Month')
    time_in_seconds = models.CharField(max_length=100, verbose_name='time_in_seconds')
    issuercountrycode = models.CharField(max_length=100, verbose_name='issuercountrycode')
    result = models.CharField(max_length=100, verbose_name='result')
    threshold = models.FloatField(default=0.0, verbose_name='threshold')

    def __str__(self):
        return f"Transaction {self.id}"


class Services (models.Model):
    name = models.CharField(max_length=100, verbose_name='name')
    desc = models.CharField(max_length=100, verbose_name='desc')
    #picture = models.CharField(max_length=100000, verbose_name='picture')
    image = models.ImageField(upload_to='services/', null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def str(self):
            return self.name
    


User = get_user_model()

class Reports(models.Model):
    FRAUD_STATUS_CHOICES = [
        ('not_fraud', 'Транзакция не мошенническая'),
        ('fraud', 'Транзакция мошенническая'),
    ]
    
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='reports',
        verbose_name='Пользователь'
    )
    name = models.CharField(max_length=255, verbose_name='Название отчёта')
    date = models.DateField(verbose_name='Дата отчёта')
    fraud_status = models.CharField(
        max_length=10, 
        choices=FRAUD_STATUS_CHOICES, 
        default='not_fraud', 
        verbose_name='Статус транзакции'
    )

    def __str__(self):
        return self.name
    

class Notification(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='notifications',
        verbose_name='Пользователь'
    )
    text = models.TextField(verbose_name='Текст уведомления')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Время создания')

    def __str__(self):
        # Выводим первые 20 символов уведомления для наглядности
        return f"Notification for {self.user.username}: {self.text[:20]}"