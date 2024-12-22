from django.urls import path
from . import views
from django.contrib.auth.views import LoginView
from django.contrib.auth.views import LogoutView



app_name = 'users'

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', LoginView.as_view(template_name='users/login.html'), name='login'),
    path('index/', views.index, name='index'),
    path('logout/', LogoutView.as_view(next_page='users:index'), name='logout'),
    path('test/', views.test, name='test'),
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('predict/', views.predict_transactions, name='predict'),
    path('admin/self-data/', views.admin_self_data, name='admin_self_data'),
]