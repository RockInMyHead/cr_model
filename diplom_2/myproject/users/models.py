from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.exceptions import ValidationError
import re
from django.utils.translation import gettext_lazy as _

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
    cvcresponsecode = models.CharField(max_length=100, verbose_name='txvariantcode')
    txvariantcode = models.CharField(max_length=100, verbose_name='bin')
    Day = models.CharField(max_length=100, verbose_name='Day')
    Month = models.CharField(max_length=100, verbose_name='Month')
    time_in_seconds = models.CharField(max_length=100, verbose_name='time_in_seconds')
    issuercountrycode = models.CharField(max_length=100, verbose_name='issuercountrycode')


class Services (models.Model):
    name = models.CharField(max_length=100, verbose_name='name')
    desc = models.CharField(max_length=100, verbose_name='desc')
    #picture = models.CharField(max_length=100000, verbose_name='picture')
    image = models.ImageField(upload_to='services/', null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def str(self):
            return self.name