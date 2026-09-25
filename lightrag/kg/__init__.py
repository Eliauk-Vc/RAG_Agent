# Storage backends bundled with this application.

STORAGE_IMPLEMENTATIONS = {'KV_STORAGE': {'implementations': ['JsonKVStorage'], 'required_methods': ['get_by_id', 'upsert']}, 'GRAPH_STORAGE': {'implementations': ['NetworkXStorage'], 'required_methods': ['upsert_node', 'upsert_edge']}, 'VECTOR_STORAGE': {'implementations': ['NanoVectorDBStorage'], 'required_methods': ['query', 'upsert']}, 'DOC_STATUS_STORAGE': {'implementations': ['JsonDocStatusStorage'], 'required_methods': ['get_docs_by_status']}}

STORAGE_ENV_REQUIREMENTS = {'JsonDocStatusStorage': [], 'JsonKVStorage': [], 'NanoVectorDBStorage': [], 'NetworkXStorage': []}

STORAGES = {'NetworkXStorage': '.kg.networkx_impl', 'JsonKVStorage': '.kg.json_kv_impl', 'NanoVectorDBStorage': '.kg.nano_vector_db_impl', 'JsonDocStatusStorage': '.kg.json_doc_status_impl'}

def verify_storage_implementation(storage_type: str, storage_name: str) -> None:
    """Verify if storage implementation is compatible with specified storage type

    Args:
        storage_type: Storage type (KV_STORAGE, GRAPH_STORAGE etc.)
        storage_name: Storage implementation name

    Raises:
        ValueError: If storage implementation is incompatible or missing required methods
    """
    if storage_type not in STORAGE_IMPLEMENTATIONS:
        raise ValueError(f'Unknown storage type: {storage_type}')
    storage_info = STORAGE_IMPLEMENTATIONS[storage_type]
    if storage_name not in storage_info['implementations']:
        raise ValueError(f"Storage implementation '{storage_name}' is not compatible with {storage_type}. Compatible implementations are: {', '.join(storage_info['implementations'])}")
