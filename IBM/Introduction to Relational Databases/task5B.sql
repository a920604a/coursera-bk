CREATE TABLE public.sales_transaction
(
    transaction_id integer,
    date date,
    time time without time zone,
    sales_outlet integer,
    staff integer,
    custom integer,
    PRIMARY KEY (transaction_id)   
);

CREATE TABLE public.product
(
    product_id integer,
    product character varying(100),
    description character varying(250),
    product_price double precision,
    product_type_id integer,
    PRIMARY KEY (product_id)
);

CREATE TABLE sales_detail
(
    seals_detail_id integer,
    transaction_id integer,
    product_id integer,
    quantity integer,
    price double precision,  
    PRIMARY KEY (seals_detail_id)  
);



CREATE TABLE public.product_type
(
    product_type_id integer,
    product_type character varying(50),
    product_category character varying(50),
    PRIMARY KEY (product_type_id)
);



ALTER TABLE public.sales_detail
    ADD FOREIGN KEY (transaction_id)
    REFERENCES public.sales_transaction (transaction_id)
    NOT VALID;


ALTER TABLE public.sales_detail
    ADD FOREIGN KEY (product_id)
    REFERENCES public.product (product_id)
    NOT VALID;


ALTER TABLE public.product
    ADD FOREIGN KEY (product_type_id)
    REFERENCES public.product_type (product_type_id)
    NOT VALID;

END;