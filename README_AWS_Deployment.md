# AWS Deployment Guide - CIFAR-10 CNN Project

Ten przewodnik opisuje jak wdrożyć projekt CIFAR-10 CNN na AWS z pełną infrastrukturą do trenowania modeli i logowania eksperymentów w MLflow.

## 🏗️ Architektura

Projekt wykorzystuje następujące usługi AWS:

- **ECS Fargate** - konteneryzacja aplikacji API i MLflow
- **Application Load Balancer** - równoważenie obciążenia
- **RDS PostgreSQL** - backend dla MLflow
- **S3** - przechowywanie artefaktów MLflow, modeli i danych treningowych
- **ECR** - rejestr kontenerów Docker
- **SageMaker** - trenowanie modeli w chmurze
- **CloudWatch** - logowanie i monitorowanie
- **Terraform** - infrastruktura jako kod

## 📋 Wymagania

### Lokalne wymagania:
- AWS CLI skonfigurowany z odpowiednimi uprawnieniami
- Docker
- Terraform >= 1.0
- Python 3.12+
- jq (do parsowania JSON)

### Uprawnienia AWS:
- AdministratorAccess (lub odpowiednie uprawnienia do ECS, RDS, S3, ECR, SageMaker, IAM)

## 🚀 Szybki start

### 1. Klonowanie i przygotowanie

```bash
git clone <repository-url>
cd 05_CIFAR_Image_Classification

# Pobierz dane CIFAR-10
mkdir -p data/raw
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O data/raw/cifar-10-python.tar.gz
```

### 2. Konfiguracja AWS

```bash
# Sprawdź konfigurację AWS
aws sts get-caller-identity

# Ustaw region (opcjonalnie)
export AWS_REGION=us-east-1
export ENVIRONMENT=dev
```

### 3. Wdrożenie infrastruktury

```bash
# Wdróż infrastrukturę AWS
chmod +x scripts/deploy_infrastructure.sh
./scripts/deploy_infrastructure.sh
```

### 4. Budowanie i wdrażanie kontenerów

```bash
# Zbuduj i wdróż obrazy Docker
chmod +x scripts/build_and_push_images.sh
./scripts/build_and_push_images.sh
```

### 5. Upload danych treningowych

```bash
# Wgraj dane CIFAR-10 do S3
chmod +x scripts/upload_data.sh
./scripts/upload_data.sh
```

### 6. Test wdrożenia

```bash
# Sprawdź status usług
aws ecs list-services --cluster cifar-cnn-cluster

# Sprawdź logi
aws logs describe-log-groups --log-group-name-prefix "/ecs/cifar-cnn"
```

## 🔧 Szczegółowa konfiguracja

### Struktura plików Terraform

```
infra/aws/terraform/
├── main.tf          # Główna konfiguracja infrastruktury
├── variables.tf     # Zmienne konfiguracyjne
├── outputs.tf       # Wartości wyjściowe
└── terraform.tfvars # Wartości zmiennych (tworzone automatycznie)
```

### Zmienne konfiguracyjne

Edytuj `terraform.tfvars` aby dostosować konfigurację:

```hcl
aws_region = "us-east-1"
environment = "dev"
vpc_cidr = "10.0.0.0/16"
rds_instance_class = "db.t3.micro"
api_cpu = 1024
api_memory = 2048
mlflow_cpu = 512
mlflow_memory = 1024
db_password = "your_secure_password"
```

### Obrazy Docker

Projekt zawiera trzy obrazy Docker:

1. **API** (`Dockerfile`) - FastAPI aplikacja
2. **MLflow** (`Dockerfile.mlflow`) - Serwer MLflow
3. **Training** (`Dockerfile.training`) - Obraz do trenowania na SageMaker

## 🎯 Trenowanie modeli

### Uruchomienie trenowania na SageMaker

```bash
# Uruchom trenowanie Base CNN
python scripts/start_training.py start --model-type base_cnn --epochs 50 --wait

# Uruchom trenowanie z innymi parametrami
python scripts/start_training.py start \
  --model-type base_cnn \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --instance-type ml.g4dn.2xlarge

# Lista aktywnych zadań treningowych
python scripts/start_training.py list
```

### Monitorowanie trenowania

1. **SageMaker Console**: https://console.aws.amazon.com/sagemaker/
2. **MLflow UI**: `http://<ALB_DNS>/mlflow`
3. **CloudWatch Logs**: `/aws/sagemaker/TrainingJobs/`

## 📊 MLflow Integration

### Konfiguracja MLflow

MLflow jest automatycznie skonfigurowany z:
- **Backend Store**: PostgreSQL (RDS)
- **Artifact Store**: S3 bucket
- **Tracking URI**: `http://<ALB_DNS>/mlflow`

### Logowanie eksperymentów

```python
import mlflow
import mlflow.tensorflow

# Ustaw tracking URI
mlflow.set_tracking_uri("http://<ALB_DNS>/mlflow")

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

## 🔍 Monitorowanie i logowanie

### CloudWatch Logs

Logi są dostępne w następujących grupach:
- `/ecs/cifar-cnn-api` - logi API
- `/ecs/cifar-cnn-mlflow` - logi MLflow
- `/aws/sagemaker/TrainingJobs/` - logi trenowania

### Health Checks

```bash
# Sprawdź zdrowie API
curl http://<ALB_DNS>/api/health

# Sprawdź zdrowie MLflow
curl http://<ALB_DNS>/mlflow/health
```

## 🔄 CI/CD Pipeline

### GitHub Actions

Pipeline automatycznie:
1. Uruchamia testy
2. Buduje obrazy Docker
3. Wdraża infrastrukturę
4. Aktualizuje usługi ECS
5. Wykonuje health checks

### Konfiguracja Secrets

W GitHub ustaw następujące secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ACCOUNT_ID`

## 🛠️ Zarządzanie

### Aktualizacja usług

```bash
# Aktualizuj usługę API
aws ecs update-service \
  --cluster cifar-cnn-cluster \
  --service cifar-cnn-api \
  --force-new-deployment

# Sprawdź status
aws ecs describe-services \
  --cluster cifar-cnn-cluster \
  --services cifar-cnn-api
```

### Skalowanie

```bash
# Zwiększ liczbę instancji API
aws ecs update-service \
  --cluster cifar-cnn-cluster \
  --service cifar-cnn-api \
  --desired-count 3
```

### Backup i przywracanie

```bash
# Backup bazy danych MLflow
aws rds create-db-snapshot \
  --db-instance-identifier cifar-cnn-mlflow-db \
  --db-snapshot-identifier mlflow-backup-$(date +%Y%m%d)
```

## 💰 Koszty

### Szacunkowe koszty (us-east-1):

- **ECS Fargate**: ~$30-50/miesiąc (2 vCPU, 4GB RAM)
- **RDS PostgreSQL**: ~$15-25/miesiąc (db.t3.micro)
- **S3**: ~$5-10/miesiąc (w zależności od użycia)
- **ALB**: ~$20/miesiąc
- **SageMaker**: Pay-per-use (trenowanie)

**Łącznie**: ~$70-105/miesiąc + koszty trenowania

## 🔧 Rozwiązywanie problemów

### Częste problemy:

1. **Błąd uprawnień AWS**
   ```bash
   aws sts get-caller-identity
   # Sprawdź czy masz odpowiednie uprawnienia
   ```

2. **Błąd połączenia z bazą danych**
   ```bash
   # Sprawdź security groups
   aws ec2 describe-security-groups --group-ids <sg-id>
   ```

3. **Błąd ładowania modelu**
   ```bash
   # Sprawdź logi ECS
   aws logs get-log-events \
     --log-group-name "/ecs/cifar-cnn-api" \
     --log-stream-name <stream-name>
   ```

4. **Błąd trenowania SageMaker**
   ```bash
   # Sprawdź logi trenowania
   aws logs describe-log-streams \
     --log-group-name "/aws/sagemaker/TrainingJobs/<job-name>"
   ```

### Logi debugowania:

```bash
# Sprawdź status Terraform
cd infra/aws/terraform
terraform show

# Sprawdź status ECS
aws ecs describe-clusters --clusters cifar-cnn-cluster

# Sprawdź status RDS
aws rds describe-db-instances --db-instance-identifier cifar-cnn-mlflow-db
```

## 📚 Dodatkowe zasoby

- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest)

## 🤝 Wsparcie

W przypadku problemów:
1. Sprawdź logi CloudWatch
2. Sprawdź status usług w AWS Console
3. Sprawdź dokumentację AWS
4. Utwórz issue w repozytorium

---

**Uwaga**: Pamiętaj o usunięciu zasobów po zakończeniu pracy, aby uniknąć niepotrzebnych kosztów:

```bash
cd infra/aws/terraform
terraform destroy -var-file="terraform.tfvars"
```
