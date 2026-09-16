# Executive Briefing: What Drives California House Prices

*Prepared 2026-08-01T14:11:00+00:00 - champion model: Ridge v001*

## Business question

Which measurable characteristics of a California census block most affect its median home value, and can we predict that value reliably enough to support portfolio, pricing, and market-entry decisions?

## The data

The analysis uses the California census housing dataset: roughly 20,600 census blocks, each described by location, housing age, room and bedroom counts, population, households, median income, coastal proximity, and the target, **median_house_value**. The data was split into training, validation, and an untouched test set; every statistic used to clean and scale the data was learned from the training portion only, so the reported accuracy is an honest estimate of performance on unseen blocks.

## Method, in one paragraph

We fit a family of transparent linear models in which every input is placed on a common scale, so each model produces one easily-read number per factor: how many dollars the predicted value moves when that factor increases by a typical amount. We compared the candidates on the validation set, selected the best performer (**Ridge**), and measured it once on the held-out test set. No opaque black-box techniques were used, so every prediction can be explained to a stakeholder in plain business terms.

## How well it predicts

- Validation error (RMSE): about $64,787
- Validation R-squared: 0.680 (share of value variation explained)
- Baseline predicted value (intercept): about $209,064

## Top 5 value drivers

1. **median_income** - District median household income (the dominant driver of value). Each one-standard-deviation move raises predicted median value by about $74,630.
2. **latitude** - North-south location (proxy for region within California). Each one-standard-deviation move lowers predicted median value by about $51,647.
3. **longitude** - East-west location (proxy for coastal/inland position). Each one-standard-deviation move lowers predicted median value by about $49,816.
4. **log_population** - Scale of population (skew-compressed). Each one-standard-deviation move lowers predicted median value by about $48,747.
5. **ocean_proximity_INLAND** - Home is inland (typically the lowest-value segment). Each one-standard-deviation move lowers predicted median value by about $27,388.

## Important caveats

- **Multicollinearity among room/household counts.** Total rooms, bedrooms, population, and households move together; their individual coefficients should be read as a group, not as independent levers.
- **Coastal nonlinearity.** The value premium for coastal proximity is not a straight line; the linear model approximates it with category indicators and will understate sharp coastal effects.
- **Census-block aggregation.** Every figure is a block-level average, not an individual house; conclusions apply to neighbourhoods, not single properties.
- **Target cap.** Median values are capped at ~$500,001 in the source census data, so the model cannot distinguish the most expensive blocks from one another and will under-predict at the very top of the market.

## Recommendations

1. Treat **median income** and **coastal proximity** as the primary levers when screening markets; they dominate predicted value.
2. Use the model for **relative** ranking of neighbourhoods rather than precise single-home valuation, given block-level aggregation and the price cap.
3. For the highest-value coastal segments, supplement the linear model with local expertise, since the linear form understates coastal premiums.
4. Revisit the model if new data lifts the value cap or adds property-level detail, which would materially improve top-of-market accuracy.
