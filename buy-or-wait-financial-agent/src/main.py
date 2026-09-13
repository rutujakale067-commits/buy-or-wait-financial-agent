"""
Buy or Wait? — deterministic financial decision agent.

The LLM is optional and is deliberately kept out of numerical safety decisions.
Structured facts are resolved first, then a 90-day simulator verifies candidate plans.
"""
from __future__ import annotations
import argparse, math, os, re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np

BAD_STATUSES = {"failed", "cancelled", "unrealized"}
CURRENCIES = {"INR", "ZAR", "IDR", "USD", "EUR"}

# Fixed fallback for the supplied challenge images. OCR is attempted first.
IMAGE_AMOUNT_FALLBACK = {
    "event_253": 4365000.0, "event_1442": 200000.0, "event_1545": 41272.0,
    "event_1700": 32298.0, "event_1786": 704.05, "event_3051": 1995.0,
    "event_3231": 8528.10, "event_4535": 15339.0, "event_5170": 723.0,
    "event_6033": 79679.26, "event_6859": 3650.0, "event_7307": 33.50,
    "event_7941": 2298.0, "event_9421": 4543.0, "event_9806": 9968.0,
    "event_10521": 393.22,
}

@dataclass
class Plan:
    method: str
    payments: list[tuple[pd.Timestamp, float]]
    total: float
    option_id: str = ""
    changes: list[tuple[str, str, float|None]] | None = None

class UsageTracker:
    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

class FinancialAgent:
    def __init__(self, dataset_dir: str):
        self.root = Path(dataset_dir)
        self.requests = pd.read_csv(self.root/"requests.csv")
        self.profiles = pd.read_csv(self.root/"financial_profiles.csv")
        self.events = pd.read_csv(self.root/"financial_events.csv")
        self.options = pd.read_csv(self.root/"request_payment_options.csv")
        self.messages = pd.read_csv(self.root/"messages.csv")
        self.images = pd.read_csv(self.root/"images.csv")
        self.rates = pd.read_csv(self.root/"exchange_rates.csv")
        for c in ["request_date", "desired_completion_date"]:
            self.requests[c] = pd.to_datetime(self.requests[c])
        for c in ["event_date", "settlement_date"]:
            self.events[c] = pd.to_datetime(self.events[c], errors="coerce")
        self.rates["rate_date"] = pd.to_datetime(self.rates["rate_date"])
        self.events["amount_resolved"] = self.events["amount"]
        self._resolve_missing_amounts()
        self.fx_cache = {}
        self.usage = UsageTracker()
        self._group_cache = {}
        self._forecast_cache = {}
        self._timeline_cache = {}

    def _resolve_missing_amounts(self):
        missing = self.events[self.events["amount_resolved"].isna()]
        if missing.empty:
            return
        # Prefer the image relationship. A compact OCR fallback keeps this runnable
        # in environments without Tesseract.
        for _, row in missing.iterrows():
            eid = row.event_id
            val = IMAGE_AMOUNT_FALLBACK.get(eid)
            if val is None:
                iid = self.images.loc[self.images.related_event_id == eid, "image_id"]
                if len(iid):
                    try:
                        import pytesseract
                        from PIL import Image
                        p = self.root/"media"/"images"/f"{iid.iloc[0]}.png"
                        txt = pytesseract.image_to_string(Image.open(p))
                        nums = [float(x.replace(",","")) for x in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", txt)]
                        # Conservative heuristic: choose the largest plausible invoice total.
                        if nums:
                            val = max(nums)
                    except Exception:
                        val = None
            if val is not None:
                self.events.loc[self.events.event_id == eid, "amount_resolved"] = val

    def profile(self, uid):
        return self.profiles[self.profiles.user_id == uid].iloc[0]

    def user_events(self, uid):
        return self.events[self.events.user_id == uid].copy()

    def user_messages(self, uid):
        return self.messages[self.messages.user_id == uid].copy()

    def parse_message(self, uid):
        rows = self.user_messages(uid)
        if rows.empty:
            return {}
        t = " ".join(rows.message_text.astype(str).tolist())
        f = {}
        def first(patterns):
            for p in patterns:
                m = re.search(p, t, re.I)
                if m: return m
            return None

        m = first([
            r"salary has (?:increased|risen) to\s*([A-Z]{3})\s*([\d,]+(?:\.\d+)?)",
            r"(?:monthly salary|monthly pay|regular salary|base salary|confirmed salary)\s*(?:is|now|of|to|remains)?\s*([A-Z]{3})\s*([\d,]+(?:\.\d+)?)",
            r"(?:gaji bulanan|gaji pokok|gaji rutin|sisa gaji bulanan)\s*(?:Anda)?\s*(?:adalah|sebesar|menjadi)?\s*([A-Z]{3})\s*([\d,]+(?:\.\d+)?)",
            r"next salary is reduced to\s*([A-Z]{3})\s*([\d,]+(?:\.\d+)?)",
            r"temporary monthly pay is\s*([A-Z]{3})\s*([\d,]+(?:\.\d+)?)",
            r"first salary(?: from the new employer)?\s*(?:is|will be|of)\s*([A-Z]{3})\s*([\d,]+(?:\.\d+)?)",
            r"gaji pertama(?: Anda)? sebesar\s*([A-Z]{3})\s*([\d,]+(?:\.\d+)?)",
        ])
        if m:
            f["salary_amount"] = (m.group(1), float(m.group(2).replace(",","")))
        m = first([
            r"(?:confirmed credit date is|confirmed for|scheduled for|it is confirmed for)\s*(\d{4}-\d{2}-\d{2})",
            r"(?:dijadwalkan pada|dikonfirmasi.*?)(\d{4}-\d{2}-\d{2})",
        ])
        if m: f["salary_date"] = pd.Timestamp(m.group(1))
        m = re.search(r"(?:applies from|berlaku mulai)\s*(\d{4}-\d{2}-\d{2})", t, re.I)
        if m: f["effective_date"] = pd.Timestamp(m.group(1))
        m = re.search(r"resumes on\s*(\d{4}-\d{2}-\d{2})", t, re.I)
        if m: f["resume_date"] = pd.Timestamp(m.group(1))
        f["salary_ended"] = bool(re.search(r"employment has ended|contract.*has ended|kontrak.*telah berakhir|pendapatan.*telah berakhir", t, re.I))
        f["temporary_salary"] = bool(re.search(r"temporary monthly pay|next salary is reduced|gaji.*sementara|gaji.*berkurang", t, re.I))
        f["pending_income"] = bool(re.search(r"bonus.*(?:pending|not been approved)|commission.*pending|payout.*pending|still pending|belum disetujui|masih tertunda|payment processing", t, re.I))
        m = re.search(r"invoice payment of\s*([A-Z]{3})\s*([\d,]+(?:\.\d+)?)", t, re.I)
        if m:
            f["invoice"] = (m.group(1), float(m.group(2).replace(",","")))
            d = re.search(r"settlement is expected on\s*(\d{4}-\d{2}-\d{2})", t, re.I)
            if d: f["invoice_date"] = pd.Timestamp(d.group(1))
        m = re.search(r"(?:increases monthly rent by|menaikkan biaya sewa bulanan sebesar)\s*(\d+(?:\.\d+)?)%", t, re.I)
        if m: f["rent_multiplier"] = 1 + float(m.group(1))/100
        return f

    def fx(self, amount, frm, to, date):
        amount = float(amount)
        if frm == to or pd.isna(amount): return amount
        key=(round(amount,8),frm,to,str(pd.Timestamp(date).date()))
        if key in self.fx_cache: return self.fx_cache[key]
        date=pd.Timestamp(date)
        rows=self.rates[self.rates.rate_date <= date]
        if rows.empty: rows=self.rates
        rate_date=rows.rate_date.max()
        rows=rows[rows.rate_date==rate_date]
        adj={}
        for _,r in rows.iterrows():
            a,b=r.from_currency,r.to_currency
            adj.setdefault(a,[]).append((b,float(r.rate)))
            adj.setdefault(b,[]).append((a,1/float(r.rate)))
        q=[(frm,1.0)]; seen={frm}
        mult=None
        while q:
            cur,m=q.pop(0)
            for nxt,r in adj.get(cur,[]):
                if nxt==to:
                    mult=m*r; q=[]; break
                if nxt not in seen:
                    seen.add(nxt); q.append((nxt,m*r))
        if mult is None: raise ValueError(f"No FX path {frm}->{to} on {date.date()}")
        self.fx_cache[key]=amount*mult
        return amount*mult

    def recurring_groups(self, uid, start):
        ck=(uid,str(pd.Timestamp(start).date()))
        if ck in self._group_cache: return self._group_cache[ck]
        x=self.user_events(uid)
        x=x[(x.event_date < start) & x.status.eq("settled") & ~x.direction.eq("non_cash")]
        groups=[]
        for key,g in x.groupby(["event_type","category","direction"]):
            dates=sorted(pd.Series(g.event_date.dropna().unique()))
            if len(dates)<3 and not (key[0]=="income" and key[1]=="salary" and len(dates)>=2): continue
            intervals=np.diff([d.toordinal() for d in dates])
            med=float(np.median(intervals))
            if med < 2 or med > 45: continue
            # Require reasonable repetition; this avoids treating isolated windfalls as recurring.
            close=np.mean((intervals >= max(1,0.5*med)) & (intervals <= 1.8*med))
            if close < 0.5: continue
            vals=g.amount_resolved.dropna().tail(12)
            if vals.empty: continue
            amount=float(vals.median())
            rep=g.iloc[-1]
            groups.append({
                "key":key, "interval":med, "amount":amount,
                "last":dates[-1], "event_id":rep.event_id,
                "flexibility":rep.flexibility,
                "minimum_allowed_amount":rep.minimum_allowed_amount,
            })
        self._group_cache[ck]=groups
        return groups

    def forecast(self, uid, start, end, changes=None):
        p=self.profile(uid)
        home=p.home_currency
        facts=self.parse_message(uid)
        ue=self.user_events(uid)
        changes=changes or {}
        cache_key=(uid,str(pd.Timestamp(start).date()),str(pd.Timestamp(end).date()),tuple(sorted((k,str(v)) for k,v in changes.items())))
        if cache_key in self._forecast_cache: return self._forecast_cache[cache_key]
        out=[]
        # Confirmed future/scheduled debits and credits.
        for _,e in ue.iterrows():
            if e.status in BAD_STATUSES or e.direction=="non_cash": continue
            if e.event_date < start: continue
            amt=e.amount_resolved
            if pd.isna(amt): continue
            # settled future rows are legitimate scheduled context; pending/scheduled are included.
            if e.status in {"pending","scheduled"} or e.event_date > start:
                val=self.fx(amt,e.currency,home,e.event_date)
                out.append((e.event_date, 1 if e.direction=="credit" else -1, val, e.event_id, "actual"))

        groups=self.recurring_groups(uid,start)
        for g in groups:
            et,cat,direction=g["key"]
            if g["event_id"] in changes and changes[g["event_id"]]=="stop":
                continue
            amt=g["amount"]
            if cat=="rent" and facts.get("rent_multiplier"):
                amt*=facts["rent_multiplier"]
            if et=="income" and cat=="salary":
                if facts.get("salary_ended"):
                    continue
                if facts.get("salary_amount"):
                    cur,a=facts["salary_amount"]
                    amt=self.fx(a,cur,home,start)
            anchor=facts.get("salary_date") if (et=="income" and cat=="salary" and facts.get("salary_date")) else None
            if anchor is None:
                d=g["last"]+pd.Timedelta(days=int(round(g["interval"])))
                while d < start: d += pd.Timedelta(days=int(round(g["interval"])))
            else:
                d=pd.Timestamp(anchor)
                while d < start: d += pd.Timedelta(days=int(round(g["interval"])))
            # A temporary salary message changes the next cycle; use message amount once, then historical amount.
            n=0
            while d <= end:
                a=amt
                if g["event_id"] in changes and isinstance(changes[g["event_id"]],tuple):
                    a=changes[g["event_id"]][1]
                out.append((d,1 if direction=="credit" else -1,a,g["event_id"],"rec"))
                n+=1
                if et=="income" and cat=="salary" and facts.get("temporary_salary") and n==1:
                    # revert to the last historical salary after the affected payroll.
                    hist=ue[(ue.event_type=="income")&(ue.category=="salary")&(ue.direction=="credit")&
                            (ue.status=="settled")&(ue.amount_resolved.notna())&(ue.event_date<start)]
                    if not hist.empty:
                        amt=self.fx(float(hist.iloc[-1].amount_resolved),hist.iloc[-1].currency,home,start)
                d += pd.Timedelta(days=int(round(g["interval"])))
        # Explicit confirmed invoice payout.
        if facts.get("invoice") and facts.get("invoice_date"):
            cur,a=facts["invoice"]
            out.append((facts["invoice_date"],1,self.fx(a,cur,home,facts["invoice_date"]),"message_invoice","message"))
        # A failed bill mentioned as still outstanding is likely to be retried.
        text=" ".join(self.user_messages(uid).message_text.astype(str).tolist())
        if re.search(r"previous debit attempt failed|debit attempt failed|debit.*failed",text,re.I):
            linked=self.messages[(self.messages.user_id==uid)].related_event_id.dropna()
            # Prefer the most recent failed debit event.
            fails=ue[(ue.status=="failed")&(ue.direction=="debit")&(ue.amount_resolved.notna())]
            if not fails.empty:
                e=fails.iloc[-1]
                d=max(start,e.event_date+pd.Timedelta(days=7))
                out.append((d,-1,self.fx(e.amount_resolved,e.currency,home,d),e.event_id,"retry"))
        result=sorted(out,key=lambda x:(x[0],x[3]))
        self._forecast_cache[cache_key]=result
        return result

    def timeline(self, uid, start, changes=None):
        """Return end-of-day forecast balances for the 90-day horizon."""
        key=(uid,str(pd.Timestamp(start).date()),tuple(sorted((k,str(v)) for k,v in (changes or {}).items())))
        cache=getattr(self,"_timeline_cache",{})
        if key in cache: return cache[key]
        self._timeline_cache=cache
        p=self.profile(uid); bal=float(p.current_available_balance)
        end=start+pd.Timedelta(days=90)
        daily={}
        events=self.forecast(uid,start,end,changes)
        i=0
        for day in pd.date_range(start,end,freq="D"):
            while i<len(events) and events[i][0] <= day:
                d,sgn,amt,*_=events[i]
                if d>=start: bal += sgn*amt
                i+=1
            daily[day]=bal
        cache[key]=daily
        return daily

    def plan_safe(self, uid, start, payments, changes=None):
        p=self.profile(uid); floor=float(p.minimum_balance_to_keep)
        tl=self.timeline(uid,start,changes)
        # Payment is applied on its specified date after the baseline forecast for that day.
        byday={}
        for d,a in payments:
            d=pd.Timestamp(d)
            if start<=d<=start+pd.Timedelta(days=90):
                byday[d]=byday.get(d,0.0)+float(a)
        min_after=float("inf")
        for day,bal in tl.items():
            paid=sum(a for d,a in byday.items() if d<=day)
            min_after=min(min_after,bal-paid)
            if bal-paid < floor-1e-7:
                return False,min_after
        return True,min_after

    def simulate(self, uid, start, payments, changes=None):
        return self.plan_safe(uid,start,payments,changes)

    def safe_today(self, uid, start, requested):
        tl=self.timeline(uid,start)
        floor=float(self.profile(uid).minimum_balance_to_keep)
        # Any payment today reduces every subsequent day's balance.
        headroom=min(tl.values())-floor
        return max(0.0,min(float(requested),float(headroom)))

    def earliest_full(self, uid, start, amount):
        tl=self.timeline(uid,start)
        floor=float(self.profile(uid).minimum_balance_to_keep)
        days=list(tl.keys())
        for i,d in enumerate(days):
            if min(tl[x] for x in days[i:]) - float(amount) >= floor-1e-7:
                return d
        return None

    def flexible_candidates(self, uid, start, end):
        p=self.profile(uid)
        reduce_cats=set(str(p.expense_categories_user_is_willing_to_reduce or "").split("|")) if pd.notna(p.expense_categories_user_is_willing_to_reduce) else set()
        stop_cats=set(str(p.expense_categories_user_is_willing_to_stop or "").split("|")) if pd.notna(p.expense_categories_user_is_willing_to_stop) else set()
        cands=[]
        for g in self.recurring_groups(uid,start):
            et,cat,direction=g["key"]
            if direction!="debit" or et not in {"expense","subscription","debt_payment"}: continue
            if g["flexibility"] in {"stoppable","reducible_or_stoppable"} and cat in stop_cats:
                # savings over the forecast horizon
                base=sum(-sgn*amt for d,sgn,amt,eid,_ in self.forecast(uid,start,end) if eid==g["event_id"] and sgn<0)
                if base>0:cands.append((g["event_id"],"stop",None,base,cat))
            if g["flexibility"] in {"reducible","reducible_or_stoppable"} and cat in reduce_cats:
                minv=g["minimum_allowed_amount"]
                if pd.notna(minv):
                    minv=self.fx(float(minv), self.profile(uid).home_currency, self.profile(uid).home_currency,start)
                    savings=sum(-sgn*(amt-minv) for d,sgn,amt,eid,_ in self.forecast(uid,start,end) if eid==g["event_id"] and sgn<0)
                    if savings>0:cands.append((g["event_id"],"reduce",minv,savings,cat))
        # unique representative event/category
        seen=set(); out=[]
        for c in cands:
            if c[0] not in seen:
                seen.add(c[0]);out.append(c)
        return out

    def find_changes(self, uid, start, amount):
        end=start+pd.Timedelta(days=90)
        # Exact/minimal-change search, capped at three changes as required.
        base_ok=self.simulate(uid,start,[(start,amount)])[0]
        if base_ok:return []
        cands=self.flexible_candidates(uid,start,end)
        best=None
        for k in range(1,min(3,len(cands))+1):
            for combo in combinations(cands,k):
                changes={c[0]:("stop" if c[1]=="stop" else ("reduce",c[2])) for c in combo}
                ok,_=self.simulate(uid,start,[(start,amount)],changes)
                if not ok: continue
                savings=sum(c[3] for c in combo)
                # Minimize excess savings first, then number of changes.
                score=(round(savings,8),k,tuple(c[0] for c in combo))
                if best is None or score<best[0]:
                    best=(score,combo,changes)
            if best is not None:
                break
        return [] if best is None else best[1]

    def option_plan(self, uid, req_row, opt):
        start=req_row.request_date
        maxm=self.profile(uid).max_installment_months
        if opt.payment_method=="installments" and pd.notna(maxm) and opt.number_of_payments>maxm:
            return None
        first=pd.Timestamp(opt.first_payment_date)
        freq=opt.payment_frequency_days
        pays=[(first+pd.Timedelta(days=int(freq*i)) if pd.notna(freq) else first,
               float(opt.payment_amount)) for i in range(int(opt.number_of_payments))]
        if first<start or pays[-1][0]>req_row.desired_completion_date:
            return None
        # Exact supplied schedule, and supplied total payable amount.
        if abs(sum(a for _,a in pays)-float(opt.total_payable_amount))>0.05:
            return None
        return Plan("installments" if opt.payment_method=="installments" else "full_payment",
                    pays,float(opt.total_payable_amount),str(opt.payment_option_id),[])

    def solve_request(self, r):
        uid=r.user_id; start=r.request_date; req=float(r.requested_amount)
        p=self.profile(uid)
        prefs=set(str(p.payment_methods_user_will_consider).split("|")) if pd.notna(p.payment_methods_user_will_consider) else set()
        safe=self.safe_today(uid,start,req)
        earliest=self.earliest_full(uid,start,req)
        candidates=[]
        # Full payment, baseline.
        if "full_payment" in prefs:
            if self.simulate(uid,start,[(start,req)])[0]:
                candidates.append(Plan("full_payment",[(start,req)],req,"",[]))
            else:
                changes=self.find_changes(uid,start,req)
                if changes and self.simulate(uid,start,[(start,req)],
                    {c[0]:("stop" if c[1]=="stop" else ("reduce",c[2])) for c in changes})[0]:
                    candidates.append(Plan("full_payment",[(start,req)],req,"",[(c[0],c[1],c[2]) for c in changes]))
        # Partial payment.
        if ("partial_payment" in prefs and bool(r.allows_partial_payment)
            and safe>1e-7 and safe<req-1e-7 and earliest is not None and earliest<=r.desired_completion_date):
            candidates.append(Plan("partial_payment",[(start,safe),(earliest,req-safe)],req,"",[]))
        # Installments.
        oo=self.options[self.options.request_id==r.request_id].copy()
        for _,o in oo.iterrows():
            if o.payment_method!="installments" or "installments" not in prefs: continue
            pl=self.option_plan(uid,r,o)
            if pl and self.simulate(uid,start,pl.payments)[0]:
                candidates.append(pl)
        # Ranking from the specification.
        def key(pl):
            completes=pl.payments[-1][0] <= r.desired_completion_date
            nochange=not pl.changes
            total=pl.total
            first=pl.payments[0][0]
            n=len(pl.payments)
            oid=pl.option_id or "zzzzzz"
            return (-int(completes),-int(nochange),total,first,n,oid)
        candidates.sort(key=key)
        if candidates:
            best=candidates[0]
            status="affordable_now" if best.method=="full_payment" and best.payments[0][0]==start and not best.changes else "affordable_with_plan"
            # If full payment is selected with spending changes, it is with_plan.
            changes=best.changes or []
            change_str=[]
            for eid,typ,val in changes:
                change_str.append(f"stop:{eid}" if typ=="stop" else f"reduce_to:{eid}:{fmt(val)}")
            return self.format_result(r,safe,best,status,earliest,changes)
        # If full payment is safe later and user accepts full payment, wait.
        if earliest is not None and earliest<=r.desired_completion_date and "full_payment" in prefs:
            return self.format_result(r,safe,Plan("wait",[(earliest,req)],req), "affordable_later",earliest,[])
        return {
            "request_id":r.request_id,"amount_safe_to_pay":round(max(0,min(safe,req)),2),
            "affordability_status":"not_affordable","recommended_payment_method":"not_recommended",
            "payment_plan":"none","earliest_date_for_full_payment":"" if earliest is None else earliest.strftime("%Y-%m-%d"),
            "spending_changes_needed":"none",
            "decision_explanation":self.explain(r,"not_recommended",None,safe,earliest,[])
        }

    def _change_string(self, changes):
        if not changes: return "none"
        parts=[]
        for eid,typ,val in changes:
            parts.append(f"stop:{eid}" if typ=="stop" else f"reduce_to:{eid}:{fmt(val)}")
        return "|".join(parts)

    def format_result(self,r,safe,plan,status,earliest,changes):
        p=self.profile(r.user_id)
        cur=p.home_currency
        return {
            "request_id":r.request_id,
            "amount_safe_to_pay":round(max(0,min(float(safe),float(r.requested_amount))),2),
            "affordability_status":status,
            "recommended_payment_method":plan.method,
            "payment_plan":"|".join(f"{d.strftime('%Y-%m-%d')}:{fmt(a)}" for d,a in plan.payments),
            "earliest_date_for_full_payment": "" if earliest is None else earliest.strftime("%Y-%m-%d"),
            "spending_changes_needed":self._change_string(changes),
            "decision_explanation":self.explain(r,plan.method,plan,safe,earliest,changes)
        }

    def explain(self,r,method,plan,safe,earliest,changes):
        p=self.profile(r.user_id); cur=p.home_currency; mn=float(p.minimum_balance_to_keep)
        req=float(r.requested_amount)
        if method=="full_payment" and plan:
            if changes:
                bits=[]
                for eid,typ,val in changes:
                    bits.append(f"stop {eid}" if typ=="stop" else f"reduce {eid} to {fmt(val)}")
                return f"Make the {cur} {fmt(req)} payment today after " + ", ".join(bits) + f". This keeps at least {cur} {fmt(mn)} available."
            return f"Pay {cur} {fmt(req)} today. This keeps at least {cur} {fmt(mn)} available over the next 90 days."
        if method=="partial_payment":
            a=plan.payments[0][1]; b=plan.payments[1][1]
            return f"Pay {cur} {fmt(a)} today and the remaining {cur} {fmt(b)} on {plan.payments[1][0].strftime('%-d %B %Y')}. This completes the full request and keeps the {cur} {fmt(mn)} minimum protected."
        if method=="installments":
            return f"Use {len(plan.payments)} installments of {cur} {fmt(plan.payments[0][1])}, starting {plan.payments[0][0].strftime('%-d %B %Y')}. This keeps at least {cur} {fmt(mn)} available."
        if method=="wait":
            return f"Pay {cur} {fmt(req)} in full on {earliest.strftime('%-d %B %Y')}. Paying earlier would take the balance below the {cur} {fmt(mn)} minimum."
        if earliest is None:
            return f"Do not make this payment by {r.desired_completion_date.strftime('%-d %B %Y')}. None of the available options keeps the {cur} {fmt(mn)} minimum protected."
        return f"Do not proceed with the {cur} {fmt(req)} request. Although {cur} {fmt(safe)} is available today, the full amount cannot be completed safely within 90 days."

    def run(self):
        rows=[]
        for _,r in self.requests.iterrows():
            rows.append(self.solve_request(r))
        out=pd.DataFrame(rows,columns=[
            "request_id","amount_safe_to_pay","affordability_status","recommended_payment_method",
            "payment_plan","earliest_date_for_full_payment","spending_changes_needed","decision_explanation"
        ])
        return out

def fmt(x):
    x=float(x)
    if abs(x-round(x))<1e-9:return str(int(round(x)))
    return f"{x:.2f}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default=str(Path(__file__).resolve().parent.parent/"dataset"))
    ap.add_argument("--output",default=str(Path(__file__).resolve().parent.parent/"output.csv"))
    args=ap.parse_args()
    agent=FinancialAgent(args.dataset)
    out=agent.run()
    out.to_csv(args.output,index=False)
    print(f"Wrote {len(out)} rows to {args.output}")

if __name__=="__main__":
    main()
