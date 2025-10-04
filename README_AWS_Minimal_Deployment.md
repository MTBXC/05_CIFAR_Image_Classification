# Minimal AWS Deployment Guide - CIFAR-10 CNN Project

Ten przewodnik opisuje jak wdrożyć projekt CIFAR-10 CNN na AWS z **minimalną konfiguracją** - tylko 3 usługi AWS zamiast 10+.

## 🎯 Minimalna architektura

**Tylko 3 usługi AWS:**
1. **EC2** - jedna instancja z wszystkimi usługami (API + MLflow)
2. **S3** - jeden bucket dla modeli i MLflow artifacts  
3. **SageMaker** - tylko do trenowania (pay-per-use)

**Funkcjonalności:**
1. **WEBAPI CIFAR-10** - FastAPI na EC2 (port 8000)
2. **MLflow** - lokalny serwer na EC2 z SQLite (port 5000)
3. **Trenowanie modeli** - SageMaker jobs (pay-per-use)

## 💰 Koszty

**Szacowany koszt: ~$15-25/miesiąc + SageMaker (tylko podczas trenowania)**

- **EC2 t3.medium**: ~$30/miesiąc
- **S3**: ~$1-5/miesiąc (w zależności od użycia)
- **SageMaker**: Pay-per-use (tylko podczas trenowania)
- **Elastic IP**: ~$3.6/miesiąc

**Oszczędność: ~$50-80/miesiąc** w porównaniu do poprzedniej konfiguracji!

## 📋 Wymagania

### Lokalne wymagania:
- AWS CLI skonfigurowany z odpowiednimi uprawnieniami
- Terraform >= 1.0
- SSH key pair
- Docker (opcjonalnie, do testowania lokalnie)

### Uprawnienia AWS:
- EC2FullAccess
- S3FullAccess  
- SageMakerFullAccess
- IAMFullAccess (do tworzenia ról)

## 🚀 Szybki start

### 1. Przygotowanie

```bash
# Przejdź do katalogu z minimalną konfiguracją
cd infra/aws/terraform-minimal

# Skopiuj przykładowy plik zmiennych
cp terraform.tfvars.example terraform.tfvars
```

### 2. Konfiguracja SSH

```bash
# Wygeneruj klucz SSH (jeśli nie masz)
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Skopiuj klucz publiczny do terraform.tfvars
cat ~/.ssh/id_rsa.pub
```

### 3. Edycja konfiguracji

Edytuj `terraform.tfvars`:

```hcl
aws_region = "us-east-1"
environment = "dev"
vpc_cidr = "10.0.0.0/16"
public_subnet_cidr = "10.0.1.0/24"

# EC2 Configuration
ec2_ami = "ami-0c02fb55956c7d316"  # Ubuntu 22.04 LTS
ec2_instance_type = "t3.medium"    # 2 vCPU, 4GB RAM
ec2_volume_size = 30               # 30GB storage

# SSH Public Key (wklej swój klucz publiczny)
ssh_public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQ... your_email@example.com"
```

### 4. Wdrożenie infrastruktury

```bash
# Inicjalizacja Terraform
terraform init

# Plan wdrożenia
terraform plan

# Wdrożenie
terraform apply
```

### 5. Konfiguracja aplikacji na EC2

```bash
# Połącz się z instancją EC2
ssh -i ~/.ssh/id_rsa ubuntu@<EC2_PUBLIC_IP>

# Sprawdź czy usługi się uruchomiły
cd /home/ubuntu/cifar-cnn
docker-compose ps

# Jeśli nie, uruchom ręcznie
docker-compose up -d

# Sprawdź logi
docker-compose logs -f
```

### 6. Test wdrożenia

```bash
# Sprawdź API
curl http://<EC2_PUBLIC_IP>:8000/api/health

# Sprawdź MLflow
curl http://<EC2_PUBLIC_IP>:5000

# Sprawdź status usług
./health_check.sh
```

## 🔧 Zarządzanie usługami

### Restart usług

```bash
# Na EC2
cd /home/ubuntu/cifar-cnn
docker-compose restart

# Lub restart całego systemu
sudo systemctl restart cifar-cnn
```

### Aktualizacja aplikacji

```bash
# Na EC2
cd /home/ubuntu/cifar-cnn
git pull  # jeśli używasz git
docker-compose build
docker-compose up -d
```

### Sprawdzanie logów

```bash
# Logi API
docker-compose logs api

# Logi MLflow
docker-compose logs mlflow

# Logi systemowe
sudo journalctl -u cifar-cnn -f
```

## 🎯 Trenowanie modeli na SageMaker

### 1. Przygotowanie danych treningowych

```bash
# Na EC2, wgraj dane do S3
aws s3 cp data/raw/cifar-10-python.tar.gz s3://<S3_BUCKET>/training-data/
```

### 2. Uruchomienie trenowania

```bash
# Na EC2
python sagemaker-training.py start \
  --s3-bucket <S3_BUCKET> \
  --model-type base_cnn \
  --epochs 50 \
  --instance-type ml.m5.large
```

### 3. Monitorowanie trenowania

```bash
# Lista zadań treningowych
python sagemaker-training.py list

# Szczegóły zadania
python sagemaker-training.py describe --job-name <JOB_NAME>

# Oczekiwanie na zakończenie
python sagemaker-training.py wait --job-name <JOB_NAME>
```

### 4. Sprawdzanie wyników w MLflow

Otwórz w przeglądarce: `http://<EC2_PUBLIC_IP>:5000`

## 📊 MLflow Integration

### Konfiguracja MLflow

MLflow jest automatycznie skonfigurowany z:
- **Backend Store**: SQLite (lokalny plik)
- **Artifact Store**: S3 bucket
- **Tracking URI**: `http://<EC2_PUBLIC_IP>:5000`

### Logowanie eksperymentów

```python
import mlflow
import mlflow.tensorflow

# Ustaw tracking URI
mlflow.set_tracking_uri("http://<EC2_PUBLIC_IP>:5000")

# Rozpocznij eksperyment
with mlflow.start_run():
    # Loguj parametry
    mlflow.log_param("epochs", 50)
    mlflow.log_param("batch_size", 32)
    
    # Loguj metryki
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.45)
    
    # Loguj model
    mlflow.tensorflow.log_model(model, "model")
```

## 🔍 Monitorowanie

### Health Checks

```bash
# Sprawdź zdrowie API
curl http://<EC2_PUBLIC_IP>:8000/api/health

# Sprawdź zdrowie MLflow
curl http://<EC2_PUBLIC_IP>:5000/health

# Sprawdź status Docker
docker-compose ps
```

### Logi systemowe

```bash
# Logi aplikacji
sudo journalctl -u cifar-cnn -f

# Logi Docker
docker-compose logs -f

# Logi systemowe
sudo tail -f /var/log/syslog
```

## 🛠️ Rozwiązywanie problemów

### Częste problemy:

1. **Błąd połączenia SSH**
   ```bash
   # Sprawdź security group
   aws ec2 describe-security-groups --group-ids <sg-id>
   
   # Sprawdź klucz SSH
   ssh -i ~/.ssh/id_rsa -v ubuntu@<EC2_PUBLIC_IP>
   ```

2. **Usługi nie uruchamiają się**
   ```bash
   # Sprawdź logi Docker
   docker-compose logs
   
   # Sprawdź status kontenerów
   docker-compose ps
   
   # Restart usług
   docker-compose restart
   ```

3. **Błąd połączenia z S3**
   ```bash
   # Sprawdź uprawnienia IAM
   aws sts get-caller-identity
   
   # Test połączenia S3
   aws s3 ls s3://<S3_BUCKET>
   ```

4. **Błąd trenowania SageMaker**
   ```bash
   # Sprawdź logi trenowania
   aws logs describe-log-streams \
     --log-group-name "/aws/sagemaker/TrainingJobs/<job-name>"
   ```

### Debugowanie:

```bash
# Sprawdź status Terraform
terraform show

# Sprawdź status EC2
aws ec2 describe-instances --instance-ids <instance-id>

# Sprawdź status S3
aws s3 ls s3://<S3_BUCKET>
```

## 🔄 Backup i przywracanie

### Backup danych

```bash
# Backup MLflow bazy danych
docker-compose exec mlflow cp /mlflow/mlflow.db /mlflow/backup-$(date +%Y%m%d).db

# Backup modeli
aws s3 sync s3://<S3_BUCKET>/models/ ./backup/models/
```

### Przywracanie

```bash
# Przywróć MLflow bazę danych
docker-compose exec mlflow cp /mlflow/backup-YYYYMMDD.db /mlflow/mlflow.db

# Przywróć modele
aws s3 sync ./backup/models/ s3://<S3_BUCKET>/models/
```

## 🗑️ Usuwanie zasobów

**UWAGA**: Pamiętaj o usunięciu zasobów po zakończeniu pracy, aby uniknąć niepotrzebnych kosztów:

```bash
# Usuń infrastrukturę
cd infra/aws/terraform-minimal
terraform destroy

# Potwierdź usunięcie wpisując 'yes'
```

## 📚 Dodatkowe zasoby

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest)

## 🤝 Wsparcie

W przypadku problemów:
1. Sprawdź logi Docker: `docker-compose logs`
2. Sprawdź status EC2 w AWS Console
3. Sprawdź dokumentację AWS
4. Utwórz issue w repozytorium

---

**Uwaga**: Ta minimalna konfiguracja jest idealna do developmentu i testowania. Dla produkcji rozważ dodanie:
- Load Balancer
- RDS zamiast SQLite
- Większej instancji EC2
- Backup strategii






















