# Nimbu Mart Vendor Offboarding and Data Destruction Policy

Document ID: NM-PROC-OFF · Version 2026.1 · Effective 1 April 2026 · Owner: Head of Procurement with Information Security
Applies to: every third-party SaaS or AI tool when a contract ends, a pilot is cancelled, or approval is withdrawn.

## 1. Access revocation

### OFF-1.1 De-provisioning on termination day
On the final day of a vendor contract or pilot, IT and Information Security must disable the SAML/OIDC SSO integration, revoke all service accounts and API tokens, and export required audit logs before workspace closure.

## 2. Data destruction

### OFF-2.1 Certificate of data destruction
For any tool that processed T2 (internal) or T3 (customer) data, Procurement must obtain a signed **Certificate of Data Destruction** from the vendor within **30 days** of contract termination confirming that all Nimbu Mart data, backups, embeddings, and cached prompts have been permanently deleted (see DP-5.4). Final invoice settlement is held until this certificate is recorded in the review file.
