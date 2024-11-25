# main.py
import os, json, time
import argparse
from create_model_customization_job import create_model_customization_job
from list_model_customization_jobs import list_model_customization_jobs  
from create_endpoint import create_endpoint
from inference import get_model_responses

if __name__ == "__main__":
    # 设置环境变量的提示
    if not os.getenv("AK") or not os.getenv("SK"):
        print("Please set the environment variables AK and SK before running the script.")
        exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output JSON file')
    args = parser.parse_args()
 
    #try:
     # fine tune
#    create_ft_result = create_model_customization_job()
#    #print (f"create_ft_result: {type(create_ft_result)}  {create_ft_result}")
#    dict_create_ft_result = json.loads(create_ft_result)
#    ft_job_id = dict_create_ft_result.get('Result', {}).get('Id')
    ft_job_id = "mcj-20241121150117-2x9zn"
    print (f"ft_job_id:{ft_job_id}")
    job_ids = [ft_job_id]

    # waiting and watch
    ft_model_id = ""
    all_phase_ts = {} # all_phase_ts:{'Preprocessing': '2024-11-21T07:01:18Z', 'Queued': '2024-11-21T07:04:17Z', 'Deploying': '2024-11-21T07:15:50Z', 'Running': '2024-11-21T07:16:16Z', 'Completing': '2024-11-21T07:23:53Z', 'Completed': '2024-11-21T07:26:08Z'}
    interval = 10
    jobs_result = {}
    ft_completed_info = {}
    while True:
        #print (f"all_phase_ts:{all_phase_ts}")
        list_jobs_result = list_model_customization_jobs(job_ids)  # 获取任务状态
        #print (f"list_jobs_result: {type(list_jobs_result)}  {list_jobs_result}")
        dict_list_jobs_result_items = json.loads(list_jobs_result).get('Result', {}).get('Items', [])
        if (len(dict_list_jobs_result_items) == 0):
            time.sleep(interval)
            continue
        ft_jobs_status = dict_list_jobs_result_items[0].get('Status', {})
        phase = ft_jobs_status.get('Phase', '')
        #print (f"phase: {phase} {type(phase)}")
        phase_ts = ft_jobs_status.get('PhaseTime', '')
        #print (f"phase_ts: {phase_ts} {type(phase_ts)}")
        if phase not in phase_ts:
            all_phase_ts[phase] = phase_ts
        if phase == 'Completed':
            print("Job completed successfully.")
            ft_completed_info = dict_list_jobs_result_items[0]
            outputs = ft_completed_info.get('Outputs', [])
            if (len(outputs) > 0):
                ft_model_id = outputs[0].get('CustomModelId', '')
                print (f"ft_model_id:{ft_model_id}")
            break
        elif phase == 'Failed':
            print("Job failed.")
            break
        else:
            time.sleep(interval)
    print (f"all_phase_ts:{all_phase_ts}")

    # create endpoint
    endpoint_name = "sxwl_scoring" 
    if (len(ft_model_id) > 0):
        create_endpoint_result = create_endpoint(endpoint_name, ft_model_id)
        endpoint_id = json.loads(create_endpoint_result).get('Result', {}).get('Id', '')
    print (f"endpoint_id:{endpoint_id}")

    # inference
    time.sleep(10)
    get_model_responses(args.input_file, args.output_file, endpoint_id)

    #except Exception as e:
    #    print(f"Error: {e}")


    # scoring



