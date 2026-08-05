import pandas as pd
from pathlib import Path

def prepare_data():
    data_dir = Path(__file__).parent
    u_data_path = data_dir / "u.data"
    u_user_path = data_dir / "u.user"
    u_item_path = data_dir / "u.item"
    out_path = data_dir / "interaction_logs.csv"

    # Load ratings (u.data)
    # user id | item id | rating | timestamp
    df = pd.read_csv(u_data_path, sep='\t', names=['candidate_id', 'job_id', 'rating', 'timestamp_unix'])
    df['timestamp'] = pd.to_datetime(df['timestamp_unix'], unit='s')
    
    # Load users (u.user)
    # user id | age | gender | occupation | zip code
    users = pd.read_csv(u_user_path, sep='|', names=['candidate_id', 'age', 'gender', 'occupation', 'zip'])
    
    # Load items (u.item)
    # movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation | Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi | Thriller | War | Western
    cols = ['job_id', 'title', 'release_date', 'video_release', 'url', 'unknown', 'Action', 'Adventure', 
            'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'FilmNoir', 
            'Horror', 'Musical', 'Mystery', 'Romance', 'SciFi', 'Thriller', 'War', 'Western']
    items = pd.read_csv(u_item_path, sep='|', encoding='latin-1', names=cols)
    
    # Merge
    df = df.merge(users, on='candidate_id', how='left')
    df = df.merge(items[['job_id', 'Action', 'Comedy', 'Drama', 'SciFi', 'Romance']], on='job_id', how='left')
    
    # Map to schema
    # rating 1-5
    # click: rating >= 3
    # shortlist: rating >= 4
    # application: rating == 5
    # relevance_grade: rating - 2 (so 0 to 3 for 2-5, and rating 1 becomes 0)
    df['click'] = (df['rating'] >= 3).astype(int)
    df['shortlist'] = (df['rating'] >= 4).astype(int)
    df['application'] = (df['rating'] == 5).astype(int)
    df['relevance_grade'] = (df['rating'] - 2).clip(lower=0).astype(int)
    
    # protected_group (e.g., gender == 'F')
    df['protected_group'] = (df['gender'] == 'F').astype(int)
    
    # Features
    # We will use age (normalized), and the genres
    df['age_norm'] = df['age'] / 100.0
    # Let's keep Action, Comedy, Drama as features
    
    df.to_csv(out_path, index=False)
    print(f"Prepared {len(df)} rows -> {out_path}")

if __name__ == "__main__":
    prepare_data()
