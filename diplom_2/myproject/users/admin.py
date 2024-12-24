from django.contrib import admin
from django.template.response import TemplateResponse
from django.urls import path
from .models import User, Role, Services, Transactions, Reports
from .forms import RegistrationForm
from django import forms
from django.utils.translation import gettext_lazy as _

class UserAdminForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['role', 'first_name', 'second_name', 'patronomyc', 'email', 'phone', 'username', 'password', 'date_of_birth']
    def __init__(self, *args, **kwargs):
        super(UserAdminForm, self).__init__(*args, **kwargs)
        self.fields['first_name'].widget.attrs['placeholder'] = 'Имя (с заглавной буквы)'
        self.fields['second_name'].widget.attrs['placeholder'] = 'Фамилия (с заглавной буквы)'
        self.fields['patronomyc'].widget.attrs['placeholder'] = 'Отчество (с заглавной буквы)'
        self.fields['phone'].widget.attrs['placeholder'] = '8-XXX-XXX-XX-XX'
        self.fields['email'].widget.attrs['placeholder'] = 'example@mail.com'
        self.fields['username'].widget.attrs['placeholder'] = 'user1@us.ru'


class CustomAdminSite(admin.AdminSite):
    site_header = _("Административная панель")
    site_title = _("Админка")
    index_title = _("Добро пожаловать в административную панель")
    print ("Site!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('self-data/', self.admin_self_data, name='admin-self-data'),
        ]
        return custom_urls + urls

    def admin_self_data(self, request):
        print("Метод admin_self_data вызван")  # Временный отладочный вывод
        if not request.user.is_authenticated or not request.user.is_staff:
            from django.http import HttpResponseForbidden
            return HttpResponseForbidden("Недостаточно прав для доступа.")
        context = {
            **self.each_context(request),
            'admin_data': request.user,
        }
        return TemplateResponse(request, "admin/base_site.html", context)


admin_site = CustomAdminSite(name="admin")

class UserAdmin(admin.ModelAdmin):
    form = UserAdminForm
    list_display = ('username','password', 'first_name', 'second_name', 'patronomyc', 'email', 'phone', 'date_joined', 'role')
    list_filter = ('date_joined', 'role')

admin_site.register(User, UserAdmin)
admin_site.register(Role)
admin_site.register(Services)
admin_site.register(Transactions)
admin_site.register(Reports)