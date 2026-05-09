set windows-powershell := true

# note: the below works only on pure Docker (not podman)
DOCKER_COMPOSE_PREFIX := if env("REMOTE", "") == "true" { "DOCKER_HOST='ssh://mcmrvls@mcmrvls.local'" } else { "" }

help:
    just --list

compose-dev-up:
    {{DOCKER_COMPOSE_PREFIX}} docker-compose -f compose.dev.yml up -d

compose-dev-up-rebuild:
    {{DOCKER_COMPOSE_PREFIX}} docker-compose -f compose.dev.yml up --build -d

compose-dev-down:
    {{DOCKER_COMPOSE_PREFIX}} docker-compose -f compose.dev.yml down -t 1

compose-dev-clean-recreate:
    just compose-dev-down && just compose-prune && just compose-dev-up-rebuild

compose-prod-up:
    docker-compose -f compose.prod.yml up -d

compose-prod-up-rebuild:
    docker-compose -f compose.prod.yml up --build -d

compose-prod-down:
    docker-compose -f compose.prod.yml down

compose-prune:
    -podman pod ps | xargs podman pod rm -f

compose-prod-clean-recreate:
    just compose-prod-down && just compose-prune && just compose-prod-up-rebuild
