#!/usr/bin/env bash

declare -A urls
urls[ucf]=https://incommon.org/custom/certificates/repository/sha384%20Intermediate%20cert.txt

for key in "${!urls[@]}"; do
  url="${urls[$key]}"
  cert_file="/usr/local/share/ca-certificates/$key.crt"

  sudo wget -qO "$cert_file" "$url"
  sudo update-ca-certificates

  # The Requests library's CA cert list also needs to be updated:
  certifi_file="$(pip show certifi | grep -oP '(?<=Location: ).*$')"/certifi/cacert.pem
  { printf "\n"; cat "$cert_file"; printf "\n"; } >> "$certifi_file"
done
