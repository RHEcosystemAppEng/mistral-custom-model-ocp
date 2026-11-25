PYTHON ?= $(firstword $(wildcard .venv/bin/python) python3)
OC ?= oc
NAMESPACE ?= mistral-custom-model-ocp
PULL_SECRET ?= ibm-entitlement
PVC_NAME ?= mistral-weights
PVC_SIZE ?= 140Gi
STORAGE_CLASS ?=
GPU_COUNT ?= 8
LAYOUT ?= .local/2411-layout.json
PREPARE_TIMEOUT ?= 14400s
SERVING_TIMEOUT ?= 3600s
REPORT_DIR ?= reports
ROUTE_URL ?=
CLEANUP_TIMEOUT ?= 300s

OPERATOR_ARGS = --namespace "$(NAMESPACE)" --pull-secret "$(PULL_SECRET)" --pvc-name "$(PVC_NAME)" --pvc-size "$(PVC_SIZE)" --storage-class "$(STORAGE_CLASS)" --gpu-count "$(GPU_COUNT)"

.PHONY: preflight inventory render-config render-deployment render-discovery render-preparation prepare install status validate restart-verify runtime-check uninstall purge

preflight:
	$(PYTHON) -m mistral_validation preflight --oc "$(OC)" $(OPERATOR_ARGS)

inventory:
	$(PYTHON) -m mistral_validation inventory --oc "$(OC)" $(OPERATOR_ARGS) --layout "$(LAYOUT)" --prepare-timeout "$(PREPARE_TIMEOUT)"

render-config:
	$(PYTHON) -m mistral_validation render-config $(OPERATOR_ARGS)

render-deployment:
	$(PYTHON) -m mistral_validation render-deployment $(OPERATOR_ARGS) --serving-timeout "$(SERVING_TIMEOUT)"

render-discovery:
	$(PYTHON) -m mistral_validation render-discovery $(OPERATOR_ARGS)

render-preparation:
	$(PYTHON) -m mistral_validation render-preparation $(OPERATOR_ARGS) --layout "$(LAYOUT)"

prepare:
	$(PYTHON) -m mistral_validation prepare --oc "$(OC)" $(OPERATOR_ARGS) --layout "$(LAYOUT)" --prepare-timeout "$(PREPARE_TIMEOUT)" --report-dir "$(REPORT_DIR)"

install:
	$(PYTHON) -m mistral_validation install --oc "$(OC)" $(OPERATOR_ARGS) --serving-timeout "$(SERVING_TIMEOUT)" --report-dir "$(REPORT_DIR)"

status:
	$(PYTHON) -m mistral_validation status --oc "$(OC)" $(OPERATOR_ARGS)

uninstall purge:
	$(PYTHON) -m mistral_validation $@ --oc "$(OC)" $(OPERATOR_ARGS) --cleanup-timeout "$(CLEANUP_TIMEOUT)"

runtime-check:
	$(PYTHON) -m mistral_validation runtime-check --oc "$(OC)" $(OPERATOR_ARGS) --serving-timeout "$(SERVING_TIMEOUT)" --report-dir "$(REPORT_DIR)"

restart-verify:
	$(PYTHON) -m mistral_validation restart-verify --oc "$(OC)" $(OPERATOR_ARGS) --serving-timeout "$(SERVING_TIMEOUT)" --report-dir "$(REPORT_DIR)"

validate:
	$(PYTHON) -m mistral_validation validate --oc "$(OC)" $(OPERATOR_ARGS) --route-url "$(ROUTE_URL)" --report-dir "$(REPORT_DIR)" --include-exchange

.PHONY: chat
ifneq ($(origin PROMPT),undefined)
# Export the literal prompt instead of expanding it in Make or in the recipe shell.
chat: export MISTRAL_CHAT_PROMPT := $(value PROMPT)
unexport PROMPT
endif
chat:
	@$(PYTHON) -m mistral_validation chat --oc "$(OC)" --namespace "$(NAMESPACE)" --route-url "$(ROUTE_URL)" --use-oc-token

.PHONY: concurrency-check
concurrency-check:
	$(PYTHON) -m mistral_validation concurrency-check --oc "$(OC)" $(OPERATOR_ARGS) --route-url "$(ROUTE_URL)" --report-dir "$(REPORT_DIR)"

.PHONY: context-check
context-check:
	$(PYTHON) -m mistral_validation context-check --oc "$(OC)" $(OPERATOR_ARGS) --route-url "$(ROUTE_URL)" --report-dir "$(REPORT_DIR)"

.PHONY: verify-run
verify-run:
	$(PYTHON) -m mistral_validation verify-run --oc "$(OC)" $(OPERATOR_ARGS) --layout "$(LAYOUT)" --prepare-timeout "$(PREPARE_TIMEOUT)" --serving-timeout "$(SERVING_TIMEOUT)" --cleanup-timeout "$(CLEANUP_TIMEOUT)" --route-url "$(ROUTE_URL)" --report-dir "$(REPORT_DIR)"

RUN_REPORT ?=
.PHONY: purge-run
purge-run:
	$(PYTHON) -m mistral_validation purge-run --oc "$(OC)" $(OPERATOR_ARGS) --run-report "$(RUN_REPORT)" --cleanup-timeout "$(CLEANUP_TIMEOUT)"

.PHONY: test
test:
	$(PYTHON) -m unittest discover -s tests -v

.PHONY: typecheck
typecheck:
	$(PYTHON) -m mypy
