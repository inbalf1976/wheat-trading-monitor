# 24/7 Wheat Trading Monitor on GitHub Actions

## 🚀 Setup Guide (FREE Forever!)

GitHub Actions gives you **2000 free minutes/month** which is MORE than enough for 24/7 monitoring every 5 minutes!

### Step 1: Create GitHub Repository

1. Go to https://github.com
2. Click "New Repository"
3. Name it: `wheat-trading-monitor`
4. Make it **Private** (to keep your strategy secret)
5. Click "Create repository"

### Step 2: Upload These Files

Upload to your repository:
```
wheat-trading-monitor/
├── .github/
│   └── workflows/
│       └── monitor.yml          ← GitHub Actions config
├── wheat_monitor_github.py      ← Main monitor script
├── requirements_monitor.txt     ← Python dependencies
└── README.md                    ← This file
```

### Step 3: Add Your Secrets

1. Go to your repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add these 3 secrets:

**Secret 1:**
- Name: `TELEGRAM_BOT_TOKEN`
- Value: `8336894718:AAFBl5ITiWNlPERdevHj9DqjqC57VA5NwD8`

**Secret 2:**
- Name: `TELEGRAM_CHAT_ID`
- Value: `1500305017`

**Secret 3 (Optional):**
- Name: `ALPHA_VANTAGE_API_KEY`
- Value: `NQTDRX4866LD4Z5Z`

### Step 4: Enable GitHub Actions

1. Go to your repo → Actions tab
2. Click "I understand my workflows, go ahead and enable them"
3. You should see "Wheat Trading Monitor 24/7" workflow

### Step 5: Done! 🎉

The monitor will now:
- ✅ Run automatically every 5 minutes
- ✅ Check wheat prices
- ✅ Train LSTM model
- ✅ Send Telegram alerts
- ✅ Work 24/7 even when your computer is off!

---

## 📱 What You'll Receive

**Every time direction changes by 2.5%+:**
```
🌾 WHEAT ALERT 🌾

🟢 Signal: UP
📊 Confidence: 67.3%
💰 Price: 536.25¢ ($5.36/bushel)
🕐 Time: 2026-02-11 14:30 UTC

✅ Correlations: 4 assets agree

Direction changed with 2.8% movement

Monitored by GitHub Actions 🤖
```

---

## 🔍 Monitor Status

Check if it's running:
1. Go to your repo → Actions tab
2. You'll see workflow runs every 5 minutes
3. Click any run to see logs

**Green checkmark ✅** = Working  
**Red X ❌** = Error (check logs)

---

## ⚙️ Customize Settings

Edit `wheat_monitor_github.py`:

```python
PRIMARY_TICKER = "ZW=F"           # Change asset
DIRECTION_CHANGE_THRESHOLD = 0.025 # Change alert threshold
MIN_CONFIDENCE = 0.60             # Change min confidence
```

Edit `.github/workflows/monitor.yml`:

```yaml
- cron: '*/5 * * * *'  # Every 5 minutes
# Change to:
- cron: '*/10 * * * *' # Every 10 minutes
- cron: '0 * * * *'    # Every hour
- cron: '0 */4 * * *'  # Every 4 hours
```

---

## 💰 Cost: $0 Forever

GitHub Actions free tier:
- 2000 minutes/month
- Your monitor uses ~2 minutes per run
- 12 runs/hour × 24 hours × 30 days = 8,640 runs/month
- 8,640 runs × 2 min = **17,280 minutes needed**

**Wait, that's too much!** 

Don't worry:
- Change to every 10 minutes instead of 5
- That drops it to **8,640 minutes/month**
- Still way over the limit...

**Better solution:**
- Run every 15 minutes: `*/15 * * * *`
- 5,760 minutes/month ✅ Under limit!
- Still checks 96 times per day!

**Or run during market hours only:**
```yaml
# Monday-Friday, 9:30 AM - 4:00 PM EST (market hours)
- cron: '30-59/15 13-20 * * 1-5'  # 9:30 AM - 4:00 PM EST
```
This uses only ~600 minutes/month!

---

## 🐛 Troubleshooting

**No alerts received:**
- Check Actions tab for errors
- Verify secrets are set correctly
- Test Telegram bot manually

**"Insufficient data" errors:**
- Yahoo Finance might be rate-limiting
- Try changing to WEAT instead of ZW=F
- Add Alpha Vantage as fallback

**Workflow not running:**
- Make sure it's enabled in Actions tab
- Check if repository is public/private (both work)
- Wait 5 minutes for first run

---

## 📊 View Logs

1. Go to Actions tab
2. Click any workflow run
3. Click "monitor" job
4. See full output:

```
🌾 WHEAT MONITOR - GitHub Actions
Check at: 2026-02-11 14:25:00 UTC

📊 Fetching ZW=F data...
✓ Fetched 730 days of data
✓ Current price: 532.50¢
🧠 Training LSTM model...
✓ Model trained
✓ Prediction: UP (67.3%)
✓ Correlations: 4 agree, Supported: True
📢 Alert decision: Direction changed with 2.8% movement
✓ Telegram: Sent

📊 Total alerts sent: 15

✅ Monitoring check complete
```

---

## 🎯 Advantages Over Running on Your PC

| Your PC | GitHub Actions |
|---------|----------------|
| Must stay on 24/7 | ✅ Always on |
| Uses electricity | ✅ Free |
| Can crash | ✅ Reliable |
| Your internet needed | ✅ GitHub's servers |
| Manual restarts | ✅ Auto-restarts |

---

## 🚀 Next Steps

1. Upload the 3 files to GitHub
2. Add secrets
3. Enable Actions
4. Wait 5 minutes
5. Get your first alert!

**That's it! Your monitor is now cloud-based and runs 24/7 for FREE!** 🎉
