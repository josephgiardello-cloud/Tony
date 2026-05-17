# TONY
Ego/stylization decay scoring via bounded proxies.

## Quickstart
```
tony --input data.json --output-format brief
tony --input data.json --config config.yaml --csv out.csv --log-level INFO
```

## Accessibility & Multi-Language Support
- All dashboard notifications and templates are WCAG/ARIA compliant.
- Multi-language support is enabled via Flask-Babel and Babel.
- To change language, use `?lang=<code>` in dashboard URLs (e.g., `/notifications?lang=es`).
- Notification content is translated using Babel message catalogs in the `locales/` directory.
- All notification and dashboard content uses ARIA roles and avoids color-only cues for accessibility.

## Compliance Features
- Audit trail logging for all notifications and actions.
- Configurable notification triggers via `notification_triggers.json`.
- Secure multi-channel delivery (email, dashboard, SMS, webhooks).
- IRS/state registry and fraud database checks integrated.

## Translation Instructions
1. Add new language catalogs in `locales/<lang_code>/LC_MESSAGES/messages.po`.
2. Use Babel CLI to update and compile translations:
   - `pybabel extract -F babel.cfg -o messages.pot .`
   - `pybabel init -i messages.pot -d locales -l <lang_code>`
   - `pybabel compile -d locales`

## Accessibility Testing
- Use browser accessibility tools to verify ARIA roles and WCAG compliance.
- All dashboard endpoints render accessible notification content.

## Notification Configuration
- Edit `notification_triggers.json` to define custom notification rules and triggers.
- Role-based and multi-channel notifications are supported.
