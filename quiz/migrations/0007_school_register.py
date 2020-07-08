# Generated by Django 3.0.1 on 2020-07-04 16:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('quiz', '0006_coordinator'),
    ]

    operations = [
        migrations.CreateModel(
            name='School_register',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('type', models.CharField(max_length=50)),
                ('name', models.CharField(max_length=150)),
                ('email', models.EmailField(max_length=154)),
                ('country', models.CharField(max_length=150)),
                ('contact', models.IntegerField()),
                ('school_name', models.CharField(max_length=150)),
                ('school_address', models.CharField(max_length=250)),
                ('school_city', models.CharField(max_length=150)),
                ('school_pincode', models.CharField(max_length=50)),
                ('school_website', models.CharField(max_length=250)),
                ('school_email', models.EmailField(max_length=254)),
                ('pname', models.CharField(max_length=150)),
                ('pmobile', models.IntegerField()),
                ('exammode', models.CharField(max_length=50)),
            ],
        ),
    ]
