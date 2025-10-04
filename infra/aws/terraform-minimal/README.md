# MLFlow Minimal Deployment on AWS

Ten katalog zawiera minimalną konfigurację Terraform do wdrożenia serwera MLFlow na AWS EC2 w regionie eu-north-1.

## Wymagania

- Terraform >= 1.0
- AWS CLI skonfigurowany z odpowiednimi uprawnieniami
- Klucz SSH (publiczny)

## Szybki start

1. **Skonfiguruj zmienne:**
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   # Edytuj terraform.tfvars i dodaj swój klucz SSH
   ```

2. **Wygeneruj klucz SSH (jeśli nie masz):**
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   # Skopiuj zawartość ~/.ssh/id_rsa.pub do terraform.tfvars
   ```

3. **Zainicjalizuj Terraform:**
   ```bash
   terraform init
   ```

4. **Sprawdź plan:**
   ```bash
   terraform plan
   ```

5. **Wdróż infrastrukturę:**
   ```bash
   terraform apply
   ```

6. **Po wdrożeniu, MLFlow będzie dostępny pod adresem:**
   ```
   http://<EC2_PUBLIC_IP>:5000
   ```

## Co zostanie utworzone

- VPC z publiczną podsiecią
- Security Group z otwartymi portami 22 (SSH) i 5000 (MLFlow)
- EC2 instance (t3.micro) z Ubuntu 22.04 LTS
- S3 bucket dla artefaktów MLFlow
- Elastic IP dla stałego adresu IP
- IAM role z uprawnieniami do S3

## Dostęp do instancji

```bash
ssh -i ~/.ssh/id_rsa ubuntu@<EC2_PUBLIC_IP>
```

## Sprawdzenie statusu

```bash
# Na instancji EC2
./health_check.sh

# Sprawdzenie kontenerów Docker
docker ps
```

## Usunięcie infrastruktury

```bash
terraform destroy
```

## Uwagi

- Instancja t3.micro kwalifikuje się do AWS Free Tier
- MLFlow używa SQLite jako backend store
- Artefakty są przechowywane w S3
- Serwer MLFlow uruchamia się automatycznie po starcie instancji














