import json
import time
import random
import uuid
import argparse
from datetime import datetime
from faker import Faker
from confluent_kafka import Producer

fake = Faker()

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "health_insurance_claims"

REGIONS = ["northeast", "northwest", "southeast", "southwest"]

PROCEDURES = {
    "99213": "office_visit_moderate",
    "99214": "office_visit_complex",
    "71046": "chest_xray",
    "93000": "ecg",
    "80053": "metabolic_panel",
    "36415": "blood_draw",
    "99395": "preventive_exam",
    "27447": "knee_replacement",
    "43239": "gastric_bypass",
    "70553": "brain_mri",
}

PROCEDURE_CODES = list(PROCEDURES.keys())


def base_claim_amount(age, bmi, smoker, children, region):
    base = 1800.0 + age * 55.0 + bmi * 35.0
    if smoker:
        base *= random.uniform(2.4, 3.2)
    base += children * 450.0
    if region in ("northeast", "northwest"):
        base *= random.uniform(1.05, 1.15)
    return round(base * random.uniform(0.78, 1.22), 2)


def generate_normal_claim():
    age = random.randint(18, 80)
    bmi = round(max(15.0, min(55.0, random.gauss(28.5, 6.2))), 1)
    smoker = random.random() < 0.195
    children = random.choices([0, 1, 2, 3, 4], weights=[35, 28, 22, 10, 5])[0]
    region = random.choice(REGIONS)
    procedure = random.choice(PROCEDURE_CODES)
    amount = base_claim_amount(age, bmi, smoker, children, region)

    return {
        "claim_id": str(uuid.uuid4()),
        "patient_id": str(uuid.uuid4()),
        "age": age,
        "sex": random.choice(["male", "female"]),
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "claim_amount": amount,
        "procedure_code": procedure,
        "procedure_description": PROCEDURES[procedure],
        "provider_id": f"PRV{random.randint(1000, 8999)}",
        "submission_timestamp": datetime.utcnow().isoformat(),
        "days_since_last_claim": random.randint(30, 730),
        "diagnosis_code": fake.bothify(text="???##"),
        "is_fraud": False,
    }


def inject_fraud_pattern_1(claim):
    claim["claim_amount"] = round(claim["claim_amount"] * random.uniform(4.2, 9.0), 2)
    claim["bmi"] = round(random.uniform(42.0, 56.0), 1)
    claim["diagnosis_code"] = "Z99.89"
    return claim


def inject_fraud_pattern_2(claim):
    claim["days_since_last_claim"] = random.randint(0, 3)
    claim["claim_amount"] = round(random.uniform(18000, 52000), 2)
    claim["procedure_code"] = random.choice(["27447", "43239", "70553"])
    claim["procedure_description"] = PROCEDURES[claim["procedure_code"]]
    return claim


def inject_fraud_pattern_3(claim):
    claim["age"] = random.randint(19, 24)
    claim["procedure_code"] = "27447"
    claim["procedure_description"] = PROCEDURES["27447"]
    claim["claim_amount"] = round(random.uniform(28000, 65000), 2)
    claim["bmi"] = round(random.uniform(22.0, 27.0), 1)
    return claim


def inject_fraud_pattern_4(claim):
    claim["provider_id"] = "PRV9999"
    claim["claim_amount"] = round(random.uniform(22000, 48000), 2)
    claim["smoker"] = True
    claim["days_since_last_claim"] = random.randint(1, 5)
    return claim


FRAUD_INJECTORS = [
    inject_fraud_pattern_1,
    inject_fraud_pattern_2,
    inject_fraud_pattern_3,
    inject_fraud_pattern_4,
]


def inject_fraud(claim):
    fn = random.choice(FRAUD_INJECTORS)
    claim = fn(claim)
    claim["is_fraud"] = True
    return claim


def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed for {msg.key()}: {err}")


def parse_args():
    parser = argparse.ArgumentParser(description="Health insurance claim Kafka producer")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=TOPIC)
    parser.add_argument("--rate", type=float, default=2.0, help="Events per second")
    parser.add_argument("--fraud-rate", type=float, default=0.08, help="Fraction of fraudulent events")
    parser.add_argument("--limit", type=int, default=0, help="Max events to produce (0 = unlimited)")
    return parser.parse_args()


def main():
    args = parse_args()

    conf = {
        "bootstrap.servers": args.bootstrap_servers,
        "queue.buffering.max.messages": 100000,
        "queue.buffering.max.ms": 200,
        "batch.num.messages": 500,
    }
    producer = Producer(conf)

    interval = 1.0 / args.rate
    count = 0

    print(f"Producer started: topic={args.topic} rate={args.rate}/s fraud_rate={args.fraud_rate}")

    try:
        while True:
            claim = generate_normal_claim()
            if random.random() < args.fraud_rate:
                claim = inject_fraud(claim)

            payload = json.dumps(claim, default=str).encode("utf-8")
            producer.produce(
                args.topic,
                key=claim["claim_id"].encode("utf-8"),
                value=payload,
                callback=delivery_report,
            )
            producer.poll(0)
            count += 1

            if count % 100 == 0:
                producer.flush()
                print(f"[{datetime.utcnow().isoformat()}] Produced {count} events")

            if args.limit > 0 and count >= args.limit:
                print(f"Reached limit of {args.limit}. Stopping.")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        producer.flush()
        print(f"Total events produced: {count}")


if __name__ == "__main__":
    main()
