# 🚀 Quick Start - AWS Deployment

Szybki przewodnik wdrożenia projektu CIFAR-10 CNN na AWS.

## ⚡ 5-minutowy start

### 1. Przygotowanie
```bash
# Pobierz dane CIFAR-10
mkdir -p data/raw
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O data/raw/cifar-10-python.tar.gz

# Ustaw zmienne środowiskowe
export AWS_REGION=us-east-1
export ENVIRONMENT=dev
```

### 2. Wdrożenie infrastruktury
```bash
chmod +x scripts/deploy_infrastructure.sh
./scripts/deploy_infrastructure.sh
```

### 3. Budowanie i wdrażanie kontenerów
```bash
chmod +x scripts/build_and_push_images.sh
./scripts/build_and_push_images.sh
```

### 4. Upload danych
```bash
chmod +x scripts/upload_data.sh
./scripts/upload_data.sh
```

### 5. Test wdrożenia
```bash
# Pobierz URL z terraform-outputs.json
ALB_DNS=$(jq -r '.alb_dns_name.value' terraform-outputs.json)

# Uruchom testy
python scripts/test_deployment.py --url "http://$ALB_DNS" --wait 120
```

## 🎯 Trenowanie modelu

```bash
# Uruchom trenowanie
python scripts/start_training.py start --model-type base_cnn --epochs 50 --wait

# Sprawdź status
python scripts/start_training.py list
```

## 🌐 Dostęp do usług

Po wdrożeniu będziesz mieć dostęp do:

- **API**: `http://<ALB_DNS>/api/`
- **MLflow UI**: `http://<ALB_DNS>/mlflow/`
- **Health Check**: `http://<ALB_DNS>/api/health`

## 🧹 Czyszczenie

```bash
chmod +x scripts/cleanup_aws.sh
./scripts/cleanup_aws.sh
```

## 📊 Monitorowanie

- **AWS Console**: https://console.aws.amazon.com/
- **SageMaker**: https://console.aws.amazon.com/sagemaker/
- **ECS**: https://console.aws.amazon.com/ecs/
- **CloudWatch**: https://console.aws.amazon.com/cloudwatch/

## 💰 Koszty

Szacunkowe koszty: ~$70-105/miesiąc + koszty trenowania

**Pamiętaj o usunięciu zasobów po zakończeniu pracy!**

---

Więcej szczegółów w [README_AWS_Deployment.md](README_AWS_Deployment.md)
