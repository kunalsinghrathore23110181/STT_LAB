from pydriller import Repository
import pandas as pd
import matplotlib.pyplot as plt
import sys

# -------------------------------
# Step 1: Collect last 500 commits
# -------------------------------
def get_commits(repo_path, last_n=500, histogram=False):
    commits = []
    count = 0
    for commit in Repository(repo_path, only_no_merge=True, order='reverse',
                             histogram_diff=histogram).traverse_commits():
        if commit.in_main_branch:
            commits.append(commit)
            count += 1
            if count == last_n:
                break
    return list(reversed(commits))  # put in correct order


# -------------------------------
# Step 2: Extract commit info
# -------------------------------
def extract_commit_info(commits, diff_type="myers"):
    rows = []
    for i, commit in enumerate(commits):
        print(f"[{i+1}/{len(commits)}] Mining commit: {commit.hash}")
        for m in commit.modified_files:
            rows.append([
                m.old_path,
                m.new_path,
                commit.hash,
                commit.parents[0] if len(commit.parents) > 0 else '',
                commit.msg,
                m.diff
            ])
    df = pd.DataFrame(rows, columns=[
        "Old File Path", "New File Path",
        "Commit SHA", "Parent Commit SHA",
        "Commit Message", f"{diff_type} Diff"
    ])
    return df


# -------------------------------
# Step 3: Main Execution
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 getCommitsInfo.py <repo_path>")
        sys.exit(1)

    repo_path = sys.argv[1]

    # Myers
    print("\n🔹 Collecting commits with Myers diff...")
    commits_myers = get_commits(repo_path, histogram=False)
    df_myers = extract_commit_info(commits_myers, "Myers")
    df_myers.to_csv("commit_differences_myers.csv", index=False)
    print("✅ Saved commit_differences_myers.csv")

    # Histogram
    print("\n🔹 Collecting commits with Histogram diff...")
    commits_hist = get_commits(repo_path, histogram=True)
    df_hist = extract_commit_info(commits_hist, "Histogram")
    df_hist.to_csv("commit_differences_hist.csv", index=False)
    print("✅ Saved commit_differences_hist.csv")

    # -------------------------------
    # Step 4: Compare Myers vs Histogram
    # -------------------------------
    df_compare = pd.DataFrame()
    df_compare["Myers Diff"] = df_myers["Myers Diff"]
    df_compare["Histogram Diff"] = df_hist["Histogram Diff"]
    df_compare["equal_diffs"] = df_compare["Myers Diff"] == df_compare["Histogram Diff"]
    df_compare["file_path"] = df_myers["New File Path"].fillna("")
    df_compare["is_code"] = df_compare["file_path"].str.endswith(".py")  # mark .py as code
    df_compare.to_csv("commit_diff_comparison.csv", index=False)
    print("✅ Saved commit_diff_comparison.csv")

    # -------------------------------
    # Step 5: Generate Charts
    # -------------------------------

        # -------------------------------
    # Step 5a: File type distribution
    # -------------------------------
    file_type_counts = df_compare["is_code"].value_counts()
    file_type_counts.index = ["Non-code", "Code"]  # rename True/False to readable labels
    file_type_counts.plot.pie(autopct='%1.1f%%')
    plt.title("File Type Distribution")
    plt.ylabel("")
    plt.show()


    # Chart 1: Equal diffs
    df_compare["equal_diffs"].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title("Equal Diffs")
    plt.ylabel("")
    plt.show()

    # Chart 2: Matches for code artifacts
    code_matches = df_compare[df_compare["is_code"]]["equal_diffs"].value_counts()
    code_matches.plot.pie(autopct='%1.1f%%')
    plt.title("Matches for code artifacts")
    plt.ylabel("")
    plt.show()

    # Chart 3: Matches for non-code artifacts
    noncode_matches = df_compare[~df_compare["is_code"]]["equal_diffs"].value_counts()
    noncode_matches.plot.pie(autopct='%1.1f%%')
    plt.title("Matches for non-code artifacts")
    plt.ylabel("")
    plt.show()

    # Chart 4: Bar chart with counts
    matches_for_code = code_matches.get(True, 0)
    non_matches_for_code = code_matches.get(False, 0)
    matches_for_noncode = noncode_matches.get(True, 0)
    non_matches_for_noncode = noncode_matches.get(False, 0)

    labels = ["Matches for code artifacts",
              "Non-matches for code artifacts",
              "Matches for non-code artifacts",
              "Non-matches for non-code artifacts"]

    values = [matches_for_code, non_matches_for_code,
              matches_for_noncode, non_matches_for_noncode]

    plt.bar(labels, values, color=["blue", "red", "blue", "red"])
    plt.xticks(rotation=20, ha="right")
    plt.title("Matches and Non-Matches for Code and Non-Code Artifacts")
    plt.ylabel("Count")
    plt.show()

    print("\n📊 Summary:")
    print(f"Matches for code artifacts: {matches_for_code}")
    print(f"Non-Matches for code artifacts: {non_matches_for_code}")
    print(f"Matches for non-code artifacts: {matches_for_noncode}")
    print(f"Non-Matches for non-code artifacts: {non_matches_for_noncode}")

