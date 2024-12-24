from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Transactions, User, Reports
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin


class RegistrationForm(UserCreationForm):
    date_of_birth = forms.DateField(
        required=True,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'placeholder': 'ДД.ММ.ГГГГ'
        }),
        label='Дата рождения'
    )
    class Meta:
        model = User
        fields = ['username', 'password1', 'password2', 'role', 'first_name','second_name','patronomyc','phone','email', 'date_of_birth']
    def __init__(self, *args, **kwargs):
        super(RegistrationForm, self).__init__(*args, **kwargs)
        self.fields['first_name'].widget.attrs['placeholder'] = 'Имя (с заглавной буквы)'
        self.fields['second_name'].widget.attrs['placeholder'] = 'Фамилия (с заглавной буквы)'
        self.fields['patronomyc'].widget.attrs['placeholder'] = 'Отчество (с заглавной буквы)'
        self.fields['phone'].widget.attrs['placeholder'] = '8-XXX-XXX-XX-XX'
        self.fields['email'].widget.attrs['placeholder'] = 'example@mail.com'
        self.fields['username'].widget.attrs['placeholder'] = 'user1@us.ru'


class Predict(forms.Form):
    bin = forms.CharField(label='BIN', max_length=100)
    amount = forms.CharField(label='Amount', max_length=100)
    shoppercountrycode = forms.CharField(label='Shopper Country Code', max_length=100)
    cardverificationcodesupplied = forms.ChoiceField(
        label='Card Verification Code Supplied',
        choices=[('yes', 'Yes'), ('no', 'No')]
    )
    cvcresponsecode = forms.ChoiceField(
        label='CVC Response Code',
        choices=[('0', '0'), ('1', '1'), ('2', '2'), ('5', '5')]
    )
    txvariantcode = forms.CharField(label='Transaction Variant Code', max_length=100)
    Day = forms.IntegerField(label='Day')
    Month = forms.IntegerField(label='Month')
    time_in_seconds = forms.FloatField(label='Time in Seconds')
    issuercountrycode = forms.CharField(label='Issuer Country Code', max_length=100)

   

class ReportsForm(forms.ModelForm):
    class Meta:
        model = Reports
        fields = ['name', 'date', 'fraud_status']
        # Поле fraud_status по умолчанию будет отображаться выпадающим списком.
        # Чтобы сделать именно радиокнопки, добавим настройку виджета:
        widgets = {
            'fraud_status': forms.RadioSelect
        }