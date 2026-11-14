# Nimbu Mart AI Vendor Incident Response Runbook

Document ID: NM-SEC-IR · Version 2026.1 · Effective 15 March 2026 · Owner: Incident Response Lead (Information Security)
Applies to: all security incidents, credential leaks, or cross-tenant exposures involving third-party AI and SaaS vendors.

## 1. Containment

### IR-1.1 Immediate containment and key rotation
When a vendor reports a security incident or Nimbu Mart detects anomalous vendor API activity, the on-call Security Engineer must **revoke or rotate all API keys, OAuth tokens, and webhook secrets** for that vendor within **2 hours** and disable the vendor's SSO application in the identity provider until triage is complete.

## 2. Regulatory notification

### IR-2.1 CERT-In and DPDP breach reporting
If an incident involves unauthorised access to customer personal data (T3) or affects system availability, Information Security and the DPO must evaluate regulatory reporting obligations: **CERT-In must be notified within 6 hours** of noticing such incidents, and the Data Protection Board of India and affected Data Principals must be notified within **72 hours** under DP-7.1 and the DPDP Act, 2023.

## 3. Recovery

### IR-3.1 Re-enabling a suspended vendor
A vendor suspended after a security incident may be re-enabled only after the vendor provides a written **post-mortem report** detailing root cause and remediation (SEC-9.3), and the Head of Information Security signs off on the recovery checklist.
