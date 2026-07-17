# Déploiement en production — VPS OVHcloud (cloud souverain 🇫🇷)

**API live : https://api-readmission.wisty.fr/docs**

Architecture : VPS OVH (Strasbourg) → UFW + fail2ban → nginx (TLS Let's Encrypt,
seule porte d'entrée publique) → conteneur Docker FastAPI lié à `127.0.0.1`.

```
Internet ──443──> nginx (TLS, reverse proxy) ──> 127.0.0.1:8000 ──> Docker: readmission-api
         ──80───> redirection 301 vers HTTPS         (loopback only)
         ──22───> sshd (clés uniquement) · fail2ban
```

## 1. Infrastructure

- **VPS-1 OVHcloud** : 2 vCores, 4 Go RAM, 40 Go NVMe, Strasbourg (SBG) —
  4,49 € HT/mois sans engagement. Ubuntu 26.04 LTS.
- **Swap 2 Go** (aucun par défaut) : filet de sécurité mémoire indispensable sur 4 Go.
  ```bash
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
  ```
- **DNS** : enregistrement `A` `api-readmission` → IP du VPS, ajouté chez le
  registrar (Squarespace) — aucun transfert de domaine nécessaire.

## 2. Durcissement SSH

Authentification **par clé uniquement**, root désactivé :

```bash
# /etc/ssh/sshd_config.d/00-hardening.conf
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
MaxAuthTries 3
```

**Piège appris en le faisant** : OpenSSH applique la *première* valeur rencontrée
(first-match-wins) et lit `sshd_config.d/` par ordre alphabétique. L'image cloud
OVH livre un `50-cloud-init.conf` avec `PasswordAuthentication yes` — un fichier
`99-hardening.conf` est donc silencieusement ignoré. D'où le préfixe `00-`.
Vérification qui fait foi : `sshd -T | grep -i passwordauthentication`.

Règle d'or appliquée : toujours garder une session SSH ouverte pendant qu'on
modifie la config SSH, et tester la reconnexion dans un second terminal.

## 3. Pare-feu & anti-intrusion

```bash
ufw default deny incoming && ufw default allow outgoing
ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443/tcp
ufw enable
apt install -y fail2ban && systemctl enable --now fail2ban
```

Constat empirique : **21 tentatives de brute-force SSH et 2 IP bannies dans la
première heure** d'existence du serveur. Le durcissement n'est pas théorique.

**Piège Docker/UFW** : Docker écrit ses propres règles iptables — un
`-p 8000:8000` expose le port publiquement *malgré* UFW. Parade systématique :
publier sur la boucle locale uniquement (`-p 127.0.0.1:8000:8000`) et ne laisser
que nginx (80/443) comme entrée publique.

## 4. Service conteneurisé

```bash
docker build -t readmission-api .           # le modèle est embarqué dans l'image
docker run -d --name readmission-api --restart unless-stopped \
  --memory=1g -p 127.0.0.1:8000:8000 \
  --env-file .env \
  -v readmission-data:/app/data \
  readmission-api
```

- `--restart unless-stopped` : relance automatique au reboot du VPS.
- `--memory=1g` : borne mémoire — le VPS héberge plusieurs services, aucun ne
  doit pouvoir affamer les autres.
- `--env-file .env` : configuration sensible (ex. `API_KEY`) injectée au runtime —
  le `.env` est exclu de l'image par le `.dockerignore` (jamais de secret cuit dans
  une couche Docker).
- `-v readmission-data:/app/data` : persiste le journal SQLite du monitoring
  (`predictions_log.db`) malgré les recréations du conteneur.
- uvicorn est lancé avec `--proxy-headers` (cf. Dockerfile) : l'IP client réelle
  (X-Forwarded-For posé par nginx) alimente le rate limiting.

## 5. Reverse proxy nginx + TLS

```nginx
# /etc/nginx/sites-available/api-readmission
server {
    listen 80;
    server_name api-readmission.wisty.fr;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
ln -s /etc/nginx/sites-available/api-readmission /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
certbot --nginx -d api-readmission.wisty.fr --redirect   # TLS + redirection 301
certbot renew --dry-run                                   # renouvellement auto (timer systemd)
```

## 6. Exploitation courante

```bash
docker ps && docker stats --no-stream        # état / conso des conteneurs
docker logs readmission-api --tail 50        # logs applicatifs
sudo fail2ban-client status sshd             # IP bannies
sudo ufw status verbose                      # règles pare-feu
df -h / && free -h                           # disque / mémoire
# Mise à jour applicative :
cd ~/readmission-risk-ml && git pull && docker build -t readmission-api . \
  && docker rm -f readmission-api \
  && docker run -d --name readmission-api --restart unless-stopped \
       --memory=1g -p 127.0.0.1:8000:8000 \
       --env-file .env \
       -v readmission-data:/app/data \
       readmission-api
```

## Coût total

4,49 € HT/mois (VPS) + 0 € (TLS, DNS inclus dans le domaine existant).
