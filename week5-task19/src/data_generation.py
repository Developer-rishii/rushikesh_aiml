import pandas as pd
import numpy as np
import os
import json

np.random.seed(42)

def generate_data(data_dir: str):
    print("Generating synthetic IRT data...")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Colleges
    num_colleges = 20
    colleges = [f"COLLEGE_{i:03d}" for i in range(1, num_colleges + 1)]
    
    # 2. Students
    num_students = 2000
    student_colleges = np.random.choice(colleges, size=num_students, p=np.random.dirichlet(np.ones(num_colleges)))
    thetas = np.random.normal(0, 1, size=num_students)
    
    students_df = pd.DataFrame({
        'student_id': [f"STU_{i:04d}" for i in range(1, num_students + 1)],
        'college_id': student_colleges,
        'true_theta': thetas
    })
    
    # 3. Items
    num_items = 500
    subjects = ["Math", "Physics", "Chemistry", "CS", "English", "History", "Biology", "Economics"]
    item_subjects = np.random.choice(subjects, size=num_items)
    
    item_colleges = []
    for _ in range(num_items):
        num_c = np.random.randint(1, 6)
        item_colleges.append(list(np.random.choice(colleges, size=num_c, replace=False)))
        
    a_params = np.zeros(num_items)
    b_params = np.zeros(num_items)
    is_weak = np.zeros(num_items, dtype=bool)
    
    for i in range(num_items):
        # 18% weak discrimination (near zero or negative)
        if np.random.rand() < 0.18:
            a_params[i] = np.random.uniform(-0.5, 0.45)
            is_weak[i] = True
        else:
            a_params[i] = np.random.uniform(0.35, 2.5)
            
        # 8% degenerate difficulty (extremely easy or extremely hard)
        if np.random.rand() < 0.08:
            b_params[i] = np.random.choice([np.random.uniform(-4, -2.5), np.random.uniform(2.5, 4)])
            is_weak[i] = True
        else:
            b_params[i] = np.random.normal(0, 1.2)

    # 70/15/15 item-level split
    splits = np.random.choice(['train', 'val', 'test'], size=num_items, p=[0.7, 0.15, 0.15])

    items_df = pd.DataFrame({
        'item_id': [f"ITM_{i:04d}" for i in range(1, num_items + 1)],
        'subject': item_subjects,
        'true_a': a_params,
        'true_b': b_params,
        'is_weak_item': is_weak,
        'allowed_colleges': item_colleges,
        'split': splits
    })
    
    # Inject item edge case: duplicate item ID
    dup_item = items_df.iloc[0].copy()
    items_df = pd.concat([items_df, pd.DataFrame([dup_item])], ignore_index=True)
    
    # 4. Responses
    responses = []
    student_dict = students_df.set_index('student_id').to_dict('index')
    item_dict_df = items_df.drop_duplicates(subset=['item_id'])
    
    response_idx = 1
    for _, item_row in item_dict_df.iterrows():
        item_id = item_row['item_id']
        a = item_row['true_a']
        b = item_row['true_b']
        allowed_c = item_row['allowed_colleges']
        
        rand_val = np.random.rand()
        if rand_val < 0.02: # Zero responses
            continue
        elif rand_val < 0.05: # Cold start
            num_responses = np.random.randint(1, 10)
        else:
            num_responses = int(np.random.exponential(400))
            num_responses = max(20, min(num_responses, num_students))
            
        eligible_students = students_df[students_df['college_id'].isin(allowed_c)]['student_id'].values
        if len(eligible_students) == 0:
            continue
        
        num_responses = min(num_responses, len(eligible_students))
        sampled_students = np.random.choice(eligible_students, size=num_responses, replace=False)
        
        for st_id in sampled_students:
            theta = student_dict[st_id]['true_theta']
            prob = 1 / (1 + np.exp(-a * (theta - b)))
            correct = int(np.random.rand() < prob)
            time_spent = np.random.lognormal(mean=np.log(45), sigma=0.5)
            
            responses.append({
                'response_id': f"R_{response_idx}",
                'student_id': st_id,
                'item_id': item_id,
                'college_id': student_dict[st_id]['college_id'],
                'correct': correct,
                'time_spent_sec': time_spent
            })
            response_idx += 1
            
    responses_df = pd.DataFrame(responses)
    
    # Inject response edge cases
    idx_single = responses_df[responses_df['item_id'] == 'ITM_0010'].index
    if not idx_single.empty:
        responses_df.loc[idx_single, 'correct'] = 1  # Single outcome (all correct)
        
    responses_df.loc[0, 'correct'] = np.nan # Malformed row (null correct)
    responses_df.loc[1, 'time_spent_sec'] = -10 # Malformed row (invalid time)
    responses_df.loc[2, 'college_id'] = np.nan # Missing college_id
    
    # Save
    students_df.to_csv(os.path.join(data_dir, "students.csv"), index=False)
    items_df['allowed_colleges'] = items_df['allowed_colleges'].apply(lambda x: ','.join(x))
    items_df.to_csv(os.path.join(data_dir, "items.csv"), index=False)
    responses_df.to_csv(os.path.join(data_dir, "responses.csv"), index=False)
    
    print(f"Generated {len(students_df)} students, {len(items_df)} items, and {len(responses_df)} responses.")

if __name__ == "__main__":
    generate_data(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
