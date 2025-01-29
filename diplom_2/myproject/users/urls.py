
from django.conf import settings
from django.urls import path
from . import views
from django.contrib.auth.views import LoginView, LogoutView
from django.conf.urls.static import static

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
    path('services/', views.services, name='services'),
    path("generated/", views.generated, name='generated'),
    path("tex/", views.tex, name='tex'),
    path("test_one/", views.test_one, name='test_one'),
    path("tex_one/", views.tex_one, name='tex_one'),
    path('update_transaction/<int:transaction_id>/', views.update_transaction, name='update_transaction'),
    path('reports/', views.reports_view, name='reports_page'),
    path('reports/download-excel/', views.download_excel_report, name='download_excel_report'),
]  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
if settings.DEBUG:  
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
