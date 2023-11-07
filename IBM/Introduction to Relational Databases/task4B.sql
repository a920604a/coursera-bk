CREATE TABLE public.sales_transaction
(
    transaction_id integer,
    date date,
    time time without time zone,
    sales_outlet integer,
    staff integer,
    customer integer,
    PRIMARY KEY (transaction_id)   
);

CREATE TABLE public.product
(
    product_id integer,
    type integer,
    product character varying(100),
    description character varying(250),
    price double precision,
    PRIMARY KEY (product_id)
);

CREATE TABLE sales_detail
(
    seals_detail_id integer,
    transaction_id integer,
    product integer,
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
