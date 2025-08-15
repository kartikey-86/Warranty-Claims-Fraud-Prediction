-- fraud_analysis.sql


CREATE INDEX IF NOT EXISTS idx_claims_dates ON claims(claim_date, purchase_date) WHERE claim_date IS NOT NULL AND purchase_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_claims_service_center_composite ON claims(service_center, claim_date, purchase_date, claim_amount);
CREATE INDEX IF NOT EXISTS idx_claims_product_fraud ON claims(product_type, fraud_label) WHERE fraud_label IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_claims_claim_id_amount ON claims(claim_id, claim_amount);
CREATE INDEX IF NOT EXISTS idx_claim_scores_optimized ON claim_scores(score DESC, claim_id, score) WHERE score IS NOT NULL;

-- Additional indexes for extended analysis
CREATE INDEX IF NOT EXISTS idx_claims_customer_dates ON claims(customer_id, claim_date, purchase_date);
CREATE INDEX IF NOT EXISTS idx_claims_amount_date ON claims(claim_amount, claim_date) WHERE claim_amount > 0;
*/


-- CONFIGURATION VARIABLES

-- Set analysis parameters (modify as needed)
SET @EARLY_FAILURE_DAYS = 14;
SET @TOP_K_INVESTIGATIONS = 500;
SET @HIGH_RISK_THRESHOLD = 0.7;
SET @MIN_CLAIMS_FOR_ANALYSIS = 10;


-- 1. DATA QUALITY ASSESSMENT

-- 1a) Basic data integrity checks
SELECT 
    'Data Quality Summary' as analysis_type,
    COUNT(*) as total_claims,
    COUNT(CASE WHEN claim_date < purchase_date THEN 1 END) as invalid_date_claims,
    COUNT(CASE WHEN claim_date IS NULL OR purchase_date IS NULL THEN 1 END) as missing_dates,
    COUNT(CASE WHEN claim_amount <= 0 OR claim_amount IS NULL THEN 1 END) as invalid_amounts,
    COUNT(CASE WHEN service_center IS NULL OR TRIM(service_center) = '' THEN 1 END) as missing_service_centers,
    ROUND(COUNT(CASE WHEN claim_date < purchase_date THEN 1 END) * 100.0 / COUNT(*), 2) as invalid_date_pct
FROM claims;

-- 1b) Detailed invalid date analysis
SELECT 
    'Invalid Date Details' as analysis_type,
    service_center,
    product_type,
    COUNT(*) as invalid_claims,
    MIN(DATEDIFF(claim_date, purchase_date)) as min_days_diff,
    MAX(DATEDIFF(claim_date, purchase_date)) as max_days_diff
FROM claims 
WHERE claim_date < purchase_date
GROUP BY service_center, product_type
ORDER BY invalid_claims DESC;


-- 2. SUSPICIOUS PATTERNS BY SERVICE CENTER


-- 2a) Enhanced service center analysis with multiple risk indicators
WITH service_center_metrics AS (
    SELECT 
        service_center,
        COUNT(*) as total_claims,
        COUNT(CASE WHEN claim_date <= DATE_ADD(purchase_date, INTERVAL @EARLY_FAILURE_DAYS DAY) THEN 1 END) as early_failures,
        AVG(claim_amount) as avg_claim_amount,
        STDDEV(claim_amount) as stddev_claim_amount,
        COUNT(DISTINCT customer_id) as unique_customers,
        COUNT(DISTINCT product_type) as product_variety,
        AVG(DATEDIFF(claim_date, purchase_date)) as avg_days_to_claim
    FROM claims 
    WHERE claim_date IS NOT NULL 
        AND purchase_date IS NOT NULL 
        AND claim_date >= purchase_date
        AND claim_amount > 0
    GROUP BY service_center
    HAVING total_claims >= @MIN_CLAIMS_FOR_ANALYSIS
)
SELECT 
    service_center,
    total_claims,
    ROUND(early_failures * 100.0 / total_claims, 2) as early_failure_pct,
    ROUND(avg_claim_amount, 2) as avg_claim_amount,
    ROUND(total_claims * 1.0 / unique_customers, 2) as claims_per_customer,
    ROUND(avg_days_to_claim, 1) as avg_days_to_claim,
    product_variety,
    -- Risk score calculation (weighted combination of factors)
    ROUND(
        (early_failures * 100.0 / total_claims) * 0.4 +  -- 40% weight on early failures
        LEAST((total_claims * 1.0 / unique_customers - 1) * 20, 40) * 0.3 +  -- 30% weight on repeat customers
        GREATEST(0, (avg_claim_amount - 1000) / 100) * 0.3,  -- 30% weight on high amounts
        2
    ) as risk_score
FROM service_center_metrics
ORDER BY risk_score DESC, early_failure_pct DESC;


-- 3. FRAUD ANALYSIS BY PRODUCT TYPE


-- 3a) Enhanced product type fraud analysis
WITH product_stats AS (
    SELECT 
        product_type,
        COUNT(*) as total_claims,
        SUM(CASE WHEN fraud_label IN ('Yes', '1', 'TRUE', 'true') THEN 1 ELSE 0 END) as confirmed_fraud,
        AVG(claim_amount) as avg_claim_amount,
        AVG(DATEDIFF(claim_date, purchase_date)) as avg_days_to_claim,
        COUNT(DISTINCT service_center) as service_centers_involved
    FROM claims 
    WHERE fraud_label IS NOT NULL
        AND claim_date IS NOT NULL 
        AND purchase_date IS NOT NULL
        AND claim_date >= purchase_date
    GROUP BY product_type
    HAVING total_claims >= @MIN_CLAIMS_FOR_ANALYSIS
)
SELECT 
    product_type,
    total_claims,
    confirmed_fraud,
    ROUND(confirmed_fraud * 100.0 / total_claims, 2) as fraud_rate_pct,
    ROUND(avg_claim_amount, 2) as avg_claim_amount,
    ROUND(avg_days_to_claim, 1) as avg_days_to_claim,
    service_centers_involved,
    ROUND(confirmed_fraud * avg_claim_amount, 2) as estimated_fraud_loss
FROM product_stats
ORDER BY fraud_rate_pct DESC, estimated_fraud_loss DESC;


-- 4. OPTIMIZED SAVINGS ESTIMATION


-- 4a) Top-K investigation targets with detailed metrics
WITH scored_claims AS (
    SELECT 
        c.claim_id,
        c.claim_amount,
        c.service_center,
        c.product_type,
        c.customer_id,
        COALESCE(cs.score, 0) as fraud_score,
        DATEDIFF(c.claim_date, c.purchase_date) as days_to_claim,
        -- Enhanced risk calculation
        CASE 
            WHEN cs.score >= @HIGH_RISK_THRESHOLD THEN 'High Risk'
            WHEN cs.score >= 0.5 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END as risk_category
    FROM claims c
    LEFT JOIN claim_scores cs ON c.claim_id = cs.claim_id
    WHERE c.claim_date IS NOT NULL 
        AND c.purchase_date IS NOT NULL
        AND c.claim_date >= c.purchase_date
        AND c.claim_amount > 0
),
top_investigations AS (
    SELECT *,
        ROW_NUMBER() OVER (ORDER BY fraud_score DESC, claim_amount DESC) as investigation_rank
    FROM scored_claims
    WHERE fraud_score > 0
)
SELECT 
    'Investigation Summary' as analysis_type,
    COUNT(*) as total_flagged_claims,
    COUNT(CASE WHEN investigation_rank <= @TOP_K_INVESTIGATIONS THEN 1 END) as recommended_investigations,
    SUM(CASE WHEN investigation_rank <= @TOP_K_INVESTIGATIONS THEN claim_amount ELSE 0 END) as estimated_savings_amount,
    ROUND(AVG(CASE WHEN investigation_rank <= @TOP_K_INVESTIGATIONS THEN fraud_score END), 3) as avg_risk_score_top_k,
    COUNT(CASE WHEN investigation_rank <= @TOP_K_INVESTIGATIONS AND risk_category = 'High Risk' THEN 1 END) as high_risk_in_top_k
FROM top_investigations;

-- 4b) Detailed breakdown of top investigation targets
SELECT 
    claim_id,
    claim_amount,
    service_center,
    product_type,
    days_to_claim,
    ROUND(fraud_score, 3) as fraud_score,
    risk_category,
    investigation_rank
FROM (
    SELECT 
        c.claim_id,
        c.claim_amount,
        c.service_center,
        c.product_type,
        DATEDIFF(c.claim_date, c.purchase_date) as days_to_claim,
        COALESCE(cs.score, 0) as fraud_score,
        CASE 
            WHEN cs.score >= @HIGH_RISK_THRESHOLD THEN 'High Risk'
            WHEN cs.score >= 0.5 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END as risk_category,
        ROW_NUMBER() OVER (ORDER BY COALESCE(cs.score, 0) DESC, c.claim_amount DESC) as investigation_rank
    FROM claims c
    LEFT JOIN claim_scores cs ON c.claim_id = cs.claim_id
    WHERE c.claim_date IS NOT NULL 
        AND c.purchase_date IS NOT NULL
        AND c.claim_date >= c.purchase_date
        AND c.claim_amount > 0
        AND COALESCE(cs.score, 0) > 0
) ranked_claims
WHERE investigation_rank <= @TOP_K_INVESTIGATIONS
ORDER BY investigation_rank;

-- 5. ADDITIONAL INSIGHTS

-- 5a) Customer behavior analysis
WITH customer_patterns AS (
    SELECT 
        customer_id,
        COUNT(*) as total_claims,
        COUNT(DISTINCT service_center) as service_centers_used,
        COUNT(DISTINCT product_type) as product_types_claimed,
        SUM(claim_amount) as total_claim_amount,
        MIN(purchase_date) as first_purchase,
        MAX(claim_date) as last_claim,
        AVG(DATEDIFF(claim_date, purchase_date)) as avg_days_to_claim
    FROM claims 
    WHERE claim_date IS NOT NULL 
        AND purchase_date IS NOT NULL
        AND claim_date >= purchase_date
    GROUP BY customer_id
    HAVING total_claims > 1
)
SELECT 
    'High-Activity Customers' as analysis_type,
    COUNT(*) as customers_with_multiple_claims,
    COUNT(CASE WHEN total_claims >= 5 THEN 1 END) as customers_with_5plus_claims,
    COUNT(CASE WHEN service_centers_used > 1 THEN 1 END) as customers_using_multiple_centers,
    ROUND(AVG(total_claim_amount), 2) as avg_total_claimed_per_customer,
    ROUND(AVG(avg_days_to_claim), 1) as avg_days_to_claim_overall
FROM customer_patterns;

-- 5b) Time-based fraud patterns
SELECT 
    YEAR(claim_date) as claim_year,
    MONTH(claim_date) as claim_month,
    COUNT(*) as total_claims,
    SUM(claim_amount) as total_amount,
    COUNT(CASE WHEN COALESCE(cs.score, 0) >= @HIGH_RISK_THRESHOLD THEN 1 END) as high_risk_claims,
    ROUND(AVG(COALESCE(cs.score, 0)), 3) as avg_fraud_score
FROM claims c
LEFT JOIN claim_scores cs ON c.claim_id = cs.claim_id
WHERE claim_date IS NOT NULL 
    AND claim_date >= DATE_SUB(CURDATE(), INTERVAL 24 MONTH)
GROUP BY YEAR(claim_date), MONTH(claim_date)
ORDER BY claim_year DESC, claim_month DESC;

-- 6. PERFORMANCE MONITORING


-- 6a) Query execution summary
SELECT 
    'Analysis Complete' as status,
    NOW() as execution_time,
    'Review all results above for comprehensive fraud insights' as note;