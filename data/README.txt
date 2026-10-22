Starcoach v16 cap300 CSV export

Files are UTF-8 CSV with a header row. Nested arrays and objects are compact JSON strings. Missing values are written as the literal null. Shop descriptions and catalogue coverage are omitted. Each shop may have up to 300 distinct products.

shops.csv: 6,476 rows; columns: address, category_names, contacts, instagram, lat, location_label, lon, name, payment_options, shop_id, supports_fast_delivery, website_url, weekly_hours
products.csv: 225,903 rows; columns: brand_name, category_name, image_urls, model, persian_title, product_id, specs
offers.csv: 331,506 rows; columns: description, last_price_changed_at, offer_id, price_toman, product_id, seller_title, shop_id, warranty
