import csv

defect_limits = {
    "Porosity": {
        "single_max": 1.5, 
        "sum_allowed": 20.0,
        "sum_not_allowed": 50.0,
        "chain_max": 25.0
    },
    "Slag Inclusion": {
        "length_allowed": 3.0,  
        "width_allowed": 0.5,
        "length_not_allowed": 5.0,
        "width_not_allowed": 1.0,
        "sum_length_allowed": 25.0,
        "sum_length_not_allowed": 50.0
    },
    "Crack": {
        "allowable": 0
    },
    "Lack of Fusion": {
        "single_allowed": 10.0,
        "sum_allowed": 25.0,
        "single_not_allowed": 25.0
    },
    "Undercut": {
        "depth_allowed": 0.5,
        "depth_not_allowed": 1.0
    }
}

def determine_status(defect_type, size_metric):
    if defect_type == "Porosity":
        d = size_metric.get("diameter", 0)
        sum_area = size_metric.get("sum_area", 0)
        chain_length = size_metric.get("chain_length", 0)
        if d > 3 or sum_area > 50 or chain_length > 25:
            return "недопустимо"
        elif d <= 1.5 and sum_area <= 20:
            return "допустимо"
        else:
            return "требующий проверки"

    elif defect_type == "Slag Inclusion":
        l = size_metric.get("length",0)
        w = size_metric.get("width",0)
        sum_length = size_metric.get("sum_length",0)
        if l > 5 or w > 1 or sum_length > 50:
            return "недопустимо"
        elif l <= 3 and w <= 0.5 and sum_length <= 25:
            return "допустимо"
        else:
            return "требующий проверки"

    elif defect_type == "Crack":
        return "недопустимо" if size_metric.get("length",0) > 0 else "допустимо"

    elif defect_type == "Lack of Fusion":
        single = size_metric.get("single_length",0)
        sum_len = size_metric.get("sum_length",0)
        if single > 25 or size_metric.get("with_other_defect",False):
            return "недопустимо"
        elif single <= 10 and sum_len <= 25:
            return "допустимо"
        else:
            return "требующий проверки"

    elif defect_type == "Undercut":
        depth = size_metric.get("depth",0)
        if depth > 1:
            return "недопустимо"
        elif depth <= 0.5:
            return "допустимо"
        else:
            return "требующий проверки"
    else:
        return "неизвестно"

def generate_csv_report(results, out_csv="report.csv"):
    """
    results: список словарей с ключами:
        - image_path
        - defect_type
        - bbox: [x_min, y_min, x_max, y_max]
        - size_metric: словарь с параметрами размера
    """
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path","defect_type","x_min","y_min","x_max","y_max","size_metric","status"])
        for r in results:
            x_min, y_min, x_max, y_max = r["bbox"]
            status = determine_status(r["defect_type"], r["size_metric"])
            writer.writerow([r["image_path"], r["defect_type"], x_min, y_min, x_max, y_max, str(r["size_metric"]), status])
    print(f"CSV отчёт сохранён в {out_csv}")
