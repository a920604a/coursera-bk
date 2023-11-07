CREATE TABLE public.sales_transaction
(
    transaction_id integer,
    date date,
    time time without time zone,
    sales_outlet integer,
    staff integer,
    customer integer,
    product integer,
    quantity integer,
    price double precision,    
    PRIMARY KEY (transaction_id)   
);

CREATE TABLE public.product
(
    product_id integer,
    category character varying(100),
    type integer,
    product character varying(100),
    description character varying(250),
    price double precision,
    PRIMARY KEY (product_id)
);