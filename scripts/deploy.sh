#!/bin/bash
# deploy.sh — Bootstrap the entire search engine on a K8s cluster.
#
# Prerequisites:
#   - kubectl configured and pointing to your cluster
#   - NGINX ingress controller installed
#   - metrics-server installed (for HPA)
#   - Prometheus Operator installed (optional, for monitoring)
#
# Usage: ./scripts/deploy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$(dirname "$SCRIPT_DIR")/k8s"

log() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $*"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}
# Optional: If REGISTRY is set (e.g. REGISTRY="ghcr.io/your-org"), build and push for production
if [ -n "${REGISTRY:-}" ]; then
    GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    IMAGE="${REGISTRY}/distributed-search-coord:${GIT_SHA}"
    
    log "Step 0/8: Building and Pushing Image -> ${IMAGE}"
    docker build -f "$(dirname "$SCRIPT_DIR")/docker/Dockerfile.coord" -t "${IMAGE}" "$(dirname "$SCRIPT_DIR")"
    docker push "${IMAGE}"
    
    # Then sed-replace the image in the YAML before apply
    echo "Updating Kubernetes manifests with new image tag..."
    sed -i.bak "s|image: distributed-search-coord:latest|image: ${IMAGE}|g" "$K8S_DIR"/coordinator-deployment.yaml
    sed -i.bak "s|image: distributed-search-coord:latest|image: ${IMAGE}|g" "$K8S_DIR"/indexer-cronjob.yaml
    rm -f "$K8S_DIR"/*.bak
fi
log "Step 1/8: Creating namespace"
kubectl apply -f "$K8S_DIR/namespace.yaml"

log "Step 2/8: Applying ConfigMap, Secrets, and PriorityClass"
kubectl apply -f "$K8S_DIR/priority-class.yaml"
kubectl apply -f "$K8S_DIR/configmap.yaml"
kubectl apply -f "$K8S_DIR/secrets.yaml"

log "Step 3/8: Creating Persistent Volume Claims"
kubectl apply -f "$K8S_DIR/pvc.yaml"

log "Step 4/8: Starting etcd"
kubectl apply -f "$K8S_DIR/etcd-statefulset.yaml"
echo "Waiting for etcd to become ready..."
kubectl wait --for=condition=ready pod -l app=etcd -n search --timeout=120s

log "Step 5/8: Starting Redis"
kubectl apply -f "$K8S_DIR/redis-deployment.yaml"
echo "Waiting for Redis to become ready..."
kubectl wait --for=condition=ready pod -l app=redis -n search --timeout=60s

log "Step 6/8: Building initial index (this may take a while)"

echo "Uploading source data to raw-data PVC..."
if [ -f "./data/wikipedia.xml" ]; then
    kubectl run data-loader -n search --image=busybox --restart=Never --labels="app=data-loader" \
        --overrides='{"spec":{"volumes":[{"name":"raw-data","persistentVolumeClaim":{"claimName":"raw-data-pvc"}}],"containers":[{"name":"data-loader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"raw-data","mountPath":"/data"}]}]}}' || true
    
    kubectl wait --for=condition=ready pod -l app=data-loader -n search --timeout=60s
    kubectl cp ./data/wikipedia.xml search/$(kubectl get pod -n search -l app=data-loader -o name | head -1 | cut -d/ -f2):/data/wikipedia.xml
    kubectl delete pod -l app=data-loader -n search
else
    echo "WARNING: ./data/wikipedia.xml not found locally. Ensure raw-data-pvc is populated manually!"
fi

kubectl apply -f "$K8S_DIR/indexer-cronjob.yaml"

echo "Deleting old initial-index job if it exists (so we start fresh)..."
kubectl delete job initial-index -n search --ignore-not-found=true
kubectl wait --for=delete job/initial-index -n search --timeout=30s 2>/dev/null || true

echo "Creating a one-off job from the CronJob..."
kubectl create job --from=cronjob/index-rebuild initial-index -n search
echo "Waiting for initial index build (timeout: 2 hours)..."
kubectl wait --for=condition=complete job/initial-index -n search --timeout=7200s || {
    echo "WARNING: Index build timed out or failed. Check logs:"
    echo "  kubectl logs -n search job/initial-index"
    echo "Continuing deployment anyway (coordinator will start with empty index)..."
}

log "Step 7/8: Starting Coordinator"
kubectl apply -f "$K8S_DIR/coordinator-deployment.yaml"
kubectl apply -f "$K8S_DIR/coordinator-hpa.yaml"
kubectl apply -f "$K8S_DIR/coordinator-pdb.yaml"
echo "Waiting for coordinator pods to become ready (this takes ~60s for ONNX model load)..."
kubectl wait --for=condition=ready pod -l app=coordinator -n search --timeout=180s

log "Step 8/8: Applying Ingress & Monitoring"
kubectl apply -f "$K8S_DIR/ingress.yaml"
kubectl apply -f "$K8S_DIR/monitoring/servicemonitor.yaml" 2>/dev/null || echo "ServiceMonitor CRD not found (Prometheus Operator not installed). Skipping."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Verify:"
echo "  kubectl get pods -n search"
echo "  kubectl get hpa -n search"
echo "  kubectl port-forward svc/coordinator -n search 8090:8090"
echo "  curl http://localhost:8090/health"
echo "  curl http://localhost:8090/cluster_stats"
echo '  curl "http://localhost:8090/search?q=machine+learning&limit=5"'
echo '  curl "http://localhost:8090/hybrid?q=database+architecture&limit=5&fusion=rrf"'
echo "  curl http://localhost:8090/metrics"
