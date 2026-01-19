"""
Delete test datasets from HuggingFace Hub.
"""

from huggingface_hub import delete_repo, HfApi


def delete_test_datasets(
    base_repo_id: str = "yulia-volkova/gpqa-diamond-test",
    dry_run: bool = True,
):
    """Delete test datasets from HuggingFace Hub."""
    
    suffixes = [
        "cue-summary",
        "base-summary",
        "cue-long",
        "base-long",
        "no-reasoning-summary",
        "no-reasoning-long",
    ]
    
    repos_to_delete = [f"{base_repo_id}-{suffix}" for suffix in suffixes]
    
    api = HfApi()
    
    print("=" * 60)
    if dry_run:
        print("DRY RUN - No repos will be deleted")
    else:
        print("DELETING REPOS")
    print("=" * 60)
    
    for repo_id in repos_to_delete:
        try:
            # Check if repo exists
            api.repo_info(repo_id=repo_id, repo_type="dataset")
            
            if dry_run:
                print(f"  [DRY RUN] Would delete: {repo_id}")
            else:
                print(f"  Deleting: {repo_id}...")
                delete_repo(repo_id=repo_id, repo_type="dataset")
                print(f"    ✓ Deleted")
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                print(f"  [SKIP] Not found: {repo_id}")
            else:
                print(f"  [ERROR] {repo_id}: {e}")
    
    print("\nDone!")
    if dry_run:
        print("\nTo actually delete, run with --confirm")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Delete test datasets from HuggingFace Hub")
    parser.add_argument(
        "--base-repo-id",
        type=str,
        default="yulia-volkova/gpqa-diamond-test",
        help="Base repo ID to delete"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete (without this flag, only shows what would be deleted)"
    )
    
    args = parser.parse_args()
    
    delete_test_datasets(
        base_repo_id=args.base_repo_id,
        dry_run=not args.confirm,
    )



