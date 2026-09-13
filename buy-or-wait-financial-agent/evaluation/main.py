import argparse, pandas as pd

REQUIRED=[
"request_id","amount_safe_to_pay","affordability_status","recommended_payment_method",
"payment_plan","earliest_date_for_full_payment","spending_changes_needed","decision_explanation"
]
ALLOWED_STATUS={"affordable_now","affordable_with_plan","affordable_later","not_affordable"}
ALLOWED_METHOD={"full_payment","partial_payment","installments","wait","not_recommended"}

def validate(output_path, requests_path):
    out=pd.read_csv(output_path).fillna("")
    req=pd.read_csv(requests_path)
    errors=[]
    if list(out.columns)!=REQUIRED: errors.append("Column order/schema mismatch")
    if len(out)!=len(req): errors.append(f"Expected {len(req)} rows, got {len(out)}")
    if out.request_id.nunique()!=len(out): errors.append("Duplicate request_id")
    if set(req.request_id)-set(out.request_id): errors.append("Missing request IDs")
    if not out.affordability_status.isin(ALLOWED_STATUS).all(): errors.append("Invalid status")
    if not out.recommended_payment_method.isin(ALLOWED_METHOD).all(): errors.append("Invalid method")
    if ((out.amount_safe_to_pay.astype(float)<-1e-6) | (out.amount_safe_to_pay.astype(float)>req.set_index("request_id").loc[out.request_id,"requested_amount"].to_numpy()+1e-6)).any():
        errors.append("amount_safe_to_pay bound violation")
    print("VALID" if not errors else "INVALID")
    for e in errors: print("-",e)
    return 0 if not errors else 1

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("requests")
    raise SystemExit(validate(ap.parse_args().output,ap.parse_args().requests))
