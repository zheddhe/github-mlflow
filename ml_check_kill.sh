#!/bin/bash

# Afficher les processus en cours sur le port 8080
lsof -i:8080

# Arrêter les processus en cours sur le port 8080
kill -9 $(lsof -t -i:8080)